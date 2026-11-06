"""Convert raw sensor depth and bimanual actions to a reusable Zarr cache."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import os
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '3D-Diffusion-Policy'))
import numpy as np
from PIL import Image
import zarr
from tqdm import tqdm
from diffusion_policy_3d.common.tcp4d import SPEC, point_cloud

def discover_episodes(data_root):
    """Read one TCP4D task regardless of its episode-name prefix."""
    source = Path(data_root).resolve()
    if (source / 'train_set').is_dir():
        source = source / 'train_set'
    if not source.is_dir():
        raise ValueError(f'{source}: dataset directory does not exist')
    episodes = sorted(
        (p for p in source.glob('*__episode_*') if p.is_dir()),
        key=lambda p: (int(p.name.rsplit('_', 1)[1]), p.name),
    )
    if len(episodes) < 2:
        raise ValueError(f'{source}: need at least two episodes')
    tasks = {json.loads((ep / 'metadata.json').read_text())['task'] for ep in episodes}
    if len(tasks) != 1:
        raise ValueError(f'{source}: expected one task, found {sorted(tasks)}')
    return source, episodes, tasks.pop()

def prepare(data_root, output, workers=4):
    if workers < 1:
        raise ValueError('workers must be positive')
    source, episodes, task = discover_episodes(data_root)
    output = Path(output).resolve()
    print(f'Task: {task}; {len(episodes)} episodes from {source}', flush=True)
    records=[]
    for ep in episodes:
        meta=json.loads((ep/'metadata.json').read_text())
        n=meta['frame_count']
        if meta['extrinsics'] != 'base_to_camera' or meta['arm_order'] != ['left','right']:
            raise ValueError(f'{ep}: unsupported metadata')
        paths=[ep/'metadata.json',ep/'intrinsic/third_person.npy',ep/'extrinsic/third_person.npy',ep/'timestamps.npy']
        for field in ['observations/joint_position.npy','action/joint_position.npy']:
            a=np.load(ep/field)
            if a.shape != (n,14) or not np.isfinite(a).all():
                raise ValueError(f'{ep}/{field}: invalid state/action')
            paths.append(ep/field)
        frames=sorted((ep/'depths/third_person').glob('*.png'))
        if [p.name for p in frames] != [f'{i:06d}.png' for i in range(n)]:
            raise ValueError(f'{ep}: missing or extra raw frames')
        paths.extend(frames)
        for p in paths:
            st=p.stat(); records.append([str(p),st.st_size,st.st_mtime_ns])
    fingerprint=hashlib.sha256(json.dumps(records).encode()).hexdigest()
    if output.exists():
        old=zarr.open_group(str(output),mode='r')
        if old.attrs.get('complete') and old.attrs.get('fingerprint')==fingerprint and old.attrs.get('preprocess')==SPEC:
            print(f'Using raw cache: {output}', flush=True); return
        raise ValueError(f'{output}: stale/incomplete cache; select a new --cache path')
    output.parent.mkdir(parents=True,exist_ok=True)
    tmp=output.with_name(output.name+'.partial')
    if tmp.exists():
        raise ValueError(f'Incomplete preparation at {tmp}; remove it explicitly before retrying')
    root=zarr.open_group(str(tmp),mode='w'); data=root.create_group('data'); meta=root.create_group('meta')
    total=sum(json.loads((p/'metadata.json').read_text())['frame_count'] for p in episodes)
    pc=data.zeros('point_cloud',shape=(total,1024,3),chunks=(128,1024,3),dtype='f4')
    state=data.zeros('state',shape=(total,14),chunks=(1024,14),dtype='f4')
    action=data.zeros('action',shape=(total,14),chunks=(1024,14),dtype='f4')
    offset=0; ends=[]; rates=[]
    # Compile once before worker threads start.
    from diffusion_policy_3d.common.tcp4d import fps
    fps(np.ones((2,3),np.float32),1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for ep in tqdm(episodes,desc='Raw-depth episodes'):
            s=np.load(ep/'observations/joint_position.npy'); n=len(s)
            k=np.load(ep/'intrinsic/third_person.npy'); e=np.load(ep/'extrinsic/third_person.npy')
            def frame(i):
                try:
                    with Image.open(ep/'depths/third_person'/f'{i:06d}.png') as im:
                        depth=np.asarray(im).copy()
                    return point_cloud(depth,k,e)
                except Exception as exc:
                    raise ValueError(f'{ep.name} frame {i}: {exc}') from exc
            pc[offset:offset+n]=np.stack(list(pool.map(frame,range(n))))
            state[offset:offset+n]=s
            action[offset:offset+n]=np.load(ep/'action/joint_position.npy')
            ts=np.load(ep/'timestamps.npy')
            if ts.shape != (n,) or not np.isfinite(ts).all() or np.any(np.diff(ts)<=0):
                raise ValueError(f'{ep}: invalid timestamps')
            rates.append(float(1/np.median(np.diff(ts))))
            offset+=n; ends.append(offset)
    meta.array('episode_ends',np.asarray(ends,dtype=np.int64))
    root.attrs.update(preprocess=SPEC,fingerprint=fingerprint,episodes=[p.name for p in episodes],
                      measured_median_hz=rates,complete=True,source=str(source),task=task)
    os.rename(tmp,output)
    print(f'Prepared {len(episodes)} episodes, {total} frames: {output}',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--data-root',required=True)
    parser.add_argument('--output',required=True); parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args(); prepare(args.data_root,args.output,args.workers)
