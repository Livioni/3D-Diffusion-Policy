"""Train DP3 on any task using the shared TCP4D dual-arm dataset format."""
import argparse
import datetime
import os
from pathlib import Path
import re
import subprocess
import sys


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--task', default=None,
                        help='Directory name under datasets, e.g. TCP4D_Pens_in_cups (default: TCP4D_Stack_bowls)')
    source.add_argument('--data-root', type=Path, help='Task directory or its train_set directory')
    parser.add_argument('--cache', type=Path, help='Default: data/<task>_raw.zarr')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()
    if min(args.batch_size, args.epochs, args.workers) < 1:
        parser.error('batch size, epochs and workers must be positive')

    data_root = args.data_root or repo / 'datasets' / (args.task or 'TCP4D_Stack_bowls')
    # Keep the task folder name before resolving a possible dataset symlink.
    task_dir = data_root.parent if data_root.name in ('train_set', 'eval_set') else data_root
    if data_root.name == 'eval_set':
        parser.error('use train_set for training; eval_set is held out')
    task_name = re.sub(r'[^a-z0-9_]+', '_', task_dir.name.lower()).strip('_')
    if (data_root / 'train_set').is_dir():
        data_root = data_root / 'train_set'
    if not data_root.is_dir():
        parser.error(f'dataset directory does not exist: {data_root}')
    cache = (args.cache or repo / 'data' / f'{task_name}_raw.zarr').resolve()
    output = (args.output_dir or repo / 'data/outputs' /
              f'{task_name}_raw_{datetime.datetime.now():%Y%m%d_%H%M%S}').resolve()
    if not args.prepare_only and (output / 'checkpoints/latest.ckpt').exists():
        parser.error('output already contains a checkpoint; choose a new output directory')

    subprocess.run([
        sys.executable, str(repo / 'scripts/prepare_tcp4d.py'),
        '--data-root', str(data_root.resolve()), '--output', str(cache),
        '--workers', str(args.workers),
    ], check=True)
    if args.prepare_only:
        return

    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = args.gpu
    env['HYDRA_FULL_ERROR'] = '1'
    env['PYTHONPATH'] = str(repo / '3D-Diffusion-Policy') + os.pathsep + env.get('PYTHONPATH', '')
    cmd = [
        sys.executable, str(repo / '3D-Diffusion-Policy/train.py'), '--config-name=tcp4d_dp3',
        f'task.name={task_name}', f'task.dataset.zarr_path={cache}', f'hydra.run.dir={output}',
        f'training.seed={args.seed}', f'training.num_epochs={args.epochs}',
        f'dataloader.batch_size={args.batch_size}', f'val_dataloader.batch_size={args.batch_size}',
        f'dataloader.num_workers={args.workers}', f'val_dataloader.num_workers={args.workers}',
    ]
    if args.smoke_test:
        cmd += ['training.num_epochs=2', 'training.max_train_steps=5', 'training.max_val_steps=3',
                'training.checkpoint_every=1', 'training.sample_every=1', 'training.lr_warmup_steps=1']
    print(f'Task: {task_name}\nCache: {cache}\nTraining output: {output}', flush=True)
    subprocess.run(cmd, env=env, check=True)


if __name__ == '__main__':
    main()
