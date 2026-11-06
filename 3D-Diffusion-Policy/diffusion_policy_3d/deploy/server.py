"""Stateless DP3 WebSocket server: observations in, predicted actions out."""
import argparse
import asyncio
import time
import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed
from diffusion_policy_3d.common.tcp4d import SPEC, point_cloud
from .protocol import pack_message, unpack_message, MAX_MESSAGE_BYTES, PROTOCOL_VERSION

OmegaConf.register_new_resolver('eval', eval, replace=True)

class Predictor:
    def __init__(self, checkpoint, device='cuda:0', seed=42, sampling_steps=None):
        payload=torch.load(checkpoint,map_location='cpu',pickle_module=dill,weights_only=False)
        self.spec=payload.get('preprocess')
        if self.spec != SPEC:
            raise ValueError('Checkpoint must contain the supported raw-depth preprocessing specification')
        self.task=OmegaConf.select(payload['cfg'], 'task.name', default='tcp4d')
        self.device=torch.device(device); self.seed=seed
        self.model=hydra.utils.instantiate(payload['cfg'].policy)
        states=payload['state_dicts']
        self.model.load_state_dict(states['ema_model'] if 'ema_model' in states else states['model'])
        self.model.to(self.device).eval().requires_grad_(False)
        if self.model.n_obs_steps != 2 or self.model.n_action_steps != 8 or self.model.action_dim != 14:
            raise ValueError('Unsupported TCP4D checkpoint dimensions')
        if sampling_steps is not None:
            if not 1<=sampling_steps<=self.model.noise_scheduler.config.num_train_timesteps:
                raise ValueError('sampling steps outside scheduler range')
            self.model.num_inference_steps=sampling_steps
        self.lock=asyncio.Lock()

    def metadata(self):
        return dict(type='metadata',protocol_version=PROTOCOL_VERSION,model='DP3-TCP4D-raw',
                    task=self.task,preprocess=self.spec,history_frames=2,action_shape=[8,14],
                    input_formats=['point_cloud+agent_pos','depths+joint_positions+intrinsics+extrinsics'],
                    point_cloud_format='float32 [T,1024,3], meters in footprint, sampled as preprocess specifies',
                    depth_format='uint16 [T,240,320], millimeters, aligned raw sensor Z depth',
                    state_layout=['left.joint1-6','left.gripper_width_m','right.joint1-6','right.gripper_width_m'],
                    action_alignment='same source row as observation; offsets 0..7, no fixed time interval',
                    seed=self.seed,max_message_bytes=MAX_MESSAGE_BYTES)

    @torch.inference_mode()
    def predict(self, request):
        started=time.perf_counter()
        if 'point_cloud' in request:
            if any(key in request for key in ('depths','joint_positions','intrinsics','extrinsics')):
                raise ValueError('choose point_cloud + agent_pos or the raw-depth input format')
            clouds=np.asarray(request['point_cloud'],dtype=np.float32)
            state=np.asarray(request['agent_pos'],dtype=np.float32)
            if clouds.ndim != 3 or clouds.shape[1:] != (self.spec['num_points'],3) or not np.isfinite(clouds).all():
                raise ValueError('point_cloud must be finite [T,1024,3], meters in footprint')
        else:
            if 'agent_pos' in request:
                raise ValueError('agent_pos requires point_cloud; use joint_positions with depths')
            depth=np.asarray(request['depths'])
            if depth.ndim != 3 or depth.shape[0] not in (1,2):
                raise ValueError('provide 1 or 2 history frames oldest first')
            clouds=np.stack([point_cloud(d,request['intrinsics'],request['extrinsics'],self.spec) for d in depth])
            state=np.asarray(request['joint_positions'],dtype=np.float32)
        if len(clouds) not in (1,2):
            raise ValueError('provide 1 or 2 history frames oldest first')
        if state.shape != (len(clouds),14) or not np.isfinite(state).all():
            raise ValueError('agent_pos / joint_positions must be finite [T,14], radians / gripper meters')
        if len(clouds)==1:
            clouds=np.repeat(clouds,2,axis=0); state=np.repeat(state,2,axis=0)
        obs=dict(point_cloud=torch.from_numpy(clouds[None]).to(self.device),
                 agent_pos=torch.from_numpy(state[None].copy()).to(self.device))
        generator=torch.Generator(device=self.device).manual_seed(self.seed)
        actions=self.model.predict_action(obs,generator=generator)['action'][0].cpu().numpy()
        if actions.shape != (8,14) or not np.isfinite(actions).all():
            raise RuntimeError('model returned invalid actions')
        return dict(type='prediction',request_id=request.get('request_id'),success=True,
                    actions=actions,inference_ms=(time.perf_counter()-started)*1000)

async def handle(ws, predictor):
    await ws.send(pack_message(predictor.metadata()))
    try:
        async for payload in ws:
            request={}
            try:
                request=unpack_message(payload)
                if request.get('request_id') is not None and type(request['request_id']) not in (str,int):
                    request = {}
                    raise ValueError('request_id must be a string or integer')
                if request.get('type')=='health':
                    response=dict(type='health',request_id=request.get('request_id'),success=True,ready=True)
                elif request.get('type')=='predict':
                    async with predictor.lock:
                        response=await asyncio.to_thread(predictor.predict,request)
                else:
                    raise ValueError('type must be predict or health')
            except (ValueError,KeyError,TypeError) as exc:
                response=dict(type='error',request_id=request.get('request_id'),success=False,
                              error=dict(code='invalid_request',message=str(exc)))
            except Exception as exc:
                response=dict(type='error',request_id=request.get('request_id'),success=False,
                              error=dict(code='inference_error',message=str(exc)))
            await ws.send(pack_message(response))
    except ConnectionClosed:
        pass

async def run(predictor,host,port):
    async with serve(lambda ws:handle(ws,predictor),host,port,max_size=MAX_MESSAGE_BYTES,compression=None):
        print(f'DP3 server ready: ws://{host}:{port} (task={predictor.task})',flush=True)
        await asyncio.Future()

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--checkpoint',required=True)
    p.add_argument('--device',default='cuda:0'); p.add_argument('--gpu',default=None)
    p.add_argument('--host',default='0.0.0.0'); p.add_argument('--port',type=int,default=8001)
    p.add_argument('--seed',type=int,default=42); p.add_argument('--sampling-steps',type=int)
    args=p.parse_args()
    if not 0<=args.seed<2**63:
        p.error('seed must be in [0,2**63)')
    predictor=Predictor(args.checkpoint,'cuda:'+args.gpu if args.gpu is not None else args.device,
                        args.seed,args.sampling_steps)
    asyncio.run(run(predictor,args.host,args.port))

if __name__=='__main__':
    main()
