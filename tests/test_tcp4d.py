import asyncio
import copy
import json
import os
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'3D-Diffusion-Policy'),str(Path(__file__).resolve().parents[1]/'scripts')]
import numpy as np
import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from diffusion_policy_3d.common.tcp4d import SPEC, point_cloud
from diffusion_policy_3d.deploy.protocol import pack_message,unpack_message,MAX_MESSAGE_BYTES

def test_known_base_transform_and_raw_holes():
    spec=copy.deepcopy(SPEC); spec.update(image_shape=[2,2],num_points=4)
    depth=np.array([[1000,0],[0,0]],np.uint16)
    transform=np.eye(4); transform[0,3]=-.5
    result=point_cloud(depth,np.eye(3),transform,spec)
    np.testing.assert_allclose(result,np.tile([.5,0,1],(4,1)))
    with pytest.raises(ValueError,match='no valid'):
        point_cloud(np.zeros_like(depth),np.eye(3),transform,spec)
    with pytest.raises(ValueError,match='uint16'):
        point_cloud(depth.astype(np.float32),np.eye(3),transform,spec)

def test_protocol_arrays_and_rejection():
    a=np.arange(24,dtype=np.uint16).reshape(2,3,4)
    np.testing.assert_array_equal(unpack_message(pack_message(dict(depths=a)))['depths'],a)
    with pytest.raises(ValueError):
        pack_message(dict(invalid=np.array([object()])))

def test_split_normalization_and_boundaries(tmp_path):
    import zarr
    from diffusion_policy_3d.dataset.tcp4d_dataset import TCP4DDataset
    root=zarr.open_group(str(tmp_path/'test.zarr'),mode='w')
    data=root.create_group('data'); meta=root.create_group('meta')
    values=np.repeat(np.arange(5,dtype=np.float32),4)
    for name,shape in [('state',(20,14)),('action',(20,14)),('point_cloud',(20,1024,3))]:
        data.array(name,np.broadcast_to(values.reshape((20,)+(1,)*(len(shape)-1)),shape).copy())
    meta.array('episode_ends',np.arange(4,21,4,dtype=np.int64)); root.attrs.update(preprocess=SPEC,complete=True)
    ds=TCP4DDataset(str(tmp_path/'test.zarr'),horizon=4,pad_before=1,pad_after=1,val_ratio=.4)
    for sampler,mask in [(ds.sampler,ds.train_mask),(ds.get_validation_dataset().sampler,ds.val_mask)]:
        for i in range(len(sampler)):
            sample=sampler.sample_sequence(i)['state']
            assert np.unique(sample).size==1
            assert mask[int(sample[0,0])]
    stats=ds.get_normalizer()['action'].get_input_stats()
    selected=np.flatnonzero(ds.train_mask)
    assert float(stats['min'][0])==selected.min()
    assert float(stats['max'][0])==selected.max()

def test_real_checkpoint_websocket():
    checkpoint=os.environ.get('TCP4D_CHECKPOINT')
    if not checkpoint:
        pytest.skip('set TCP4D_CHECKPOINT to run GPU integration')
    from diffusion_policy_3d.deploy.server import Predictor,handle
    from client_tcp4d import episode_request
    repo=Path(__file__).resolve().parents[1]
    episode=os.environ.get('TCP4D_EPISODE',repo/'datasets/TCP4D_Pens_in_cups/eval_set/place_pen_in_cups__episode_70')
    request=episode_request(episode,1)
    predictor=Predictor(checkpoint)
    expected=predictor.predict(request)
    direct=dict(type='predict',request_id='point-cloud',
                point_cloud=np.stack([point_cloud(d,request['intrinsics'],request['extrinsics']) for d in request['depths']]),
                agent_pos=request['joint_positions'])
    async def verify():
        async with serve(lambda ws:handle(ws,predictor),'127.0.0.1',0,max_size=MAX_MESSAGE_BYTES,compression=None) as server:
            port=server.sockets[0].getsockname()[1]
            async with connect(f'ws://127.0.0.1:{port}',max_size=MAX_MESSAGE_BYTES,compression=None,proxy=None) as ws:
                assert unpack_message(await ws.recv())['preprocess']['depth_source']=='raw'
                for bad in [dict(request,depths=np.zeros_like(request['depths'])),dict(request,joint_positions=np.zeros((2,13))),dict(request,depths=request['depths'].astype(np.float32))]:
                    await ws.send(pack_message(bad)); response=unpack_message(await ws.recv())
                    assert not response['success'] and 'actions' not in response
                await ws.send(pack_message(dict(type='health',request_id='health')))
                assert unpack_message(await ws.recv())['ready']
                await ws.send(pack_message(request)); response=unpack_message(await ws.recv())
                assert response['success'] and response['actions'].shape==(8,14)
                np.testing.assert_allclose(response['actions'],expected['actions'],rtol=1e-5,atol=1e-6)
                await ws.send(pack_message(direct)); response=unpack_message(await ws.recv())
                assert response['success'] and response['request_id']=='point-cloud'
                np.testing.assert_allclose(response['actions'],expected['actions'],rtol=1e-5,atol=1e-6)
                for bad in [dict(direct,point_cloud=np.zeros((2,1023,3))),
                            dict(direct,agent_pos=np.full((2,14),np.nan)),
                            dict(direct,depths=request['depths'])]:
                    await ws.send(pack_message(bad)); assert not unpack_message(await ws.recv())['success']
                short=dict(request,depths=request['depths'][-1:],joint_positions=request['joint_positions'][-1:])
                await ws.send(pack_message(short)); assert unpack_message(await ws.recv())['success']
                short=dict(direct,point_cloud=direct['point_cloud'][-1:],agent_pos=direct['agent_pos'][-1:])
                await ws.send(pack_message(short)); assert unpack_message(await ws.recv())['success']
                print('raw_depth_inference_ms',response['inference_ms'])
    asyncio.run(verify())


@pytest.mark.parametrize('task', ['stack_bowls', 'place_pen_in_cups', 'another_task'])
def test_prepare_shared_format_and_cache_isolation(tmp_path, task):
    from PIL import Image
    import zarr
    from prepare_tcp4d import discover_episodes, prepare

    source=tmp_path/task/'train_set'
    for number in [10,2]:
        episode=source/f'{task}__episode_{number}'
        for folder in ['observations','action','intrinsic','extrinsic','depths/third_person']:
            (episode/folder).mkdir(parents=True)
        metadata=dict(task=task,frame_count=2,extrinsics='base_to_camera',arm_order=['left','right'])
        (episode/'metadata.json').write_text(json.dumps(metadata))
        for field in ['observations','action']:
            np.save(episode/field/'joint_position.npy',np.full((2,14),number,dtype=np.float32))
        np.save(episode/'timestamps.npy',np.array([0.,.1]))
        np.save(episode/'intrinsic/third_person.npy',np.array([[300,0,160],[0,300,120],[0,0,1]]))
        transform=np.eye(4); transform[0,3]=-.6
        np.save(episode/'extrinsic/third_person.npy',transform)
        for frame in range(2):
            Image.fromarray(np.full((240,320),1000,dtype=np.uint16)).save(episode/'depths/third_person'/f'{frame:06d}.png')
    # An unrelated calibration file must not be treated as an episode.
    (source/'calibration.yaml').touch()
    _,episodes,discovered=discover_episodes(source.parent)
    assert discovered==task and [p.name for p in episodes]==[f'{task}__episode_2',f'{task}__episode_10']
    cache=tmp_path/f'{task}.zarr'
    prepare(source.parent,cache,workers=1)
    root=zarr.open_group(str(cache),mode='r')
    assert root.attrs['task']==task and root.attrs['complete']
    np.testing.assert_array_equal(root['meta/episode_ends'][:],[2,4])
    np.testing.assert_array_equal(root['data/state'][:,0],[2,2,10,10])
    np.testing.assert_array_equal(root['data/action'][:,0],[2,2,10,10])
    assert np.isfinite(root['data/point_cloud'][:]).all()
    prepare(source,cache,workers=1)
    # Changing source task metadata must prevent reuse of the old cache.
    for episode in episodes:
        metadata['task']='different_task'
        (episode/'metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='stale/incomplete'):
        prepare(source,cache,workers=1)
    metadata['task']=task
    (episodes[0]/'metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='expected one task'):
        discover_episodes(source)


@pytest.mark.parametrize('task', ['TCP4D_Stack_bowls', 'TCP4D_Pens_in_cups'])
def test_launcher_task_specific_defaults(monkeypatch, tmp_path, task):
    import train_tcp4d
    source=tmp_path/task/'train_set'
    source.mkdir(parents=True)
    calls=[]
    monkeypatch.setattr(sys,'argv',['train_tcp4d.py','--data-root',str(source),
                                 '--output-dir',str(tmp_path/'output')])
    monkeypatch.setattr(train_tcp4d.subprocess,'run',lambda cmd,**kwargs:calls.append(cmd))
    train_tcp4d.main()
    assert Path(calls[0][calls[0].index('--output')+1]).name==f'{task.lower()}_raw.zarr'
    assert f'task.name={task.lower()}' in calls[1]

def test_best_validation_checkpoint(tmp_path):
    from train import TrainDP3Workspace
    from omegaconf import OmegaConf
    workspace=TrainDP3Workspace.__new__(TrainDP3Workspace)
    workspace._output_dir=str(tmp_path)
    workspace.cfg=OmegaConf.create(dict(checkpoint=dict(topk=dict(monitor_key='val_loss',mode='min'))))
    (tmp_path/'checkpoints').mkdir()
    for name in ['latest.ckpt','epoch=0000-val_loss=0.201947.ckpt','epoch=0001-val_loss=0.188956.ckpt']:
        (tmp_path/'checkpoints'/name).touch()
    assert workspace.get_checkpoint_path('best').name=='epoch=0001-val_loss=0.188956.ckpt'
