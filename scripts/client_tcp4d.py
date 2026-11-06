"""Dataset example client; real callers send aligned raw sensor depth and measured joints."""
import argparse
import asyncio
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'3D-Diffusion-Policy'))
import numpy as np
from PIL import Image
from websockets.asyncio.client import connect
from diffusion_policy_3d.deploy.protocol import pack_message, unpack_message, MAX_MESSAGE_BYTES

def episode_request(episode,frame):
    episode=Path(episode); indices=[max(0,frame-1),frame]
    depths=[]
    for i in indices:
        with Image.open(episode/'depths/third_person'/f'{i:06d}.png') as im:
            depths.append(np.asarray(im).copy())
    return dict(type='predict',request_id=f'frame-{frame}',depths=np.stack(depths),
                joint_positions=np.load(episode/'observations/joint_position.npy')[indices],
                intrinsics=np.load(episode/'intrinsic/third_person.npy'),
                extrinsics=np.load(episode/'extrinsic/third_person.npy'))

async def call(url,request):
    async with connect(url,max_size=MAX_MESSAGE_BYTES,compression=None,proxy=None) as ws:
        metadata=unpack_message(await ws.recv())
        if metadata['preprocess']['depth_source']!='raw':
            raise ValueError('expected raw depth service')
        await ws.send(pack_message(request))
        result=unpack_message(await ws.recv())
        if not result['success']:
            raise RuntimeError(result['error'])
        return result

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--episode',type=Path,required=True)
    p.add_argument('--frame',type=int,default=1); p.add_argument('--url',default='ws://127.0.0.1:8001')
    p.add_argument('--output',type=Path,default=Path('data/tcp4d_prediction.npz'))
    args=p.parse_args(); request=episode_request(args.episode,args.frame)
    result=asyncio.run(call(args.url,request)); args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez(args.output,actions=result['actions'],inference_ms=result['inference_ms'])
    print('actions:',result['actions'].shape,'inference_ms:',round(result['inference_ms'],2),'saved:',args.output)
