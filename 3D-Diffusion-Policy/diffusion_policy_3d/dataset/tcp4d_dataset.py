import copy
import numpy as np
import torch
import zarr
from diffusion_policy_3d.common.tcp4d import SPEC
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer

class TCP4DDataset(BaseDataset):
    def __init__(self, zarr_path, horizon=16, pad_before=1, pad_after=7, seed=42, val_ratio=.1):
        root = zarr.open_group(zarr_path, mode='r')
        if root.attrs.get('preprocess') != SPEC or not root.attrs.get('complete'):
            raise ValueError('Expected complete TCP4D raw-depth cache; run preparation first')
        self.preprocess = dict(root.attrs['preprocess'])
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=['state','action','point_cloud'])
        self.val_mask = get_val_mask(self.replay_buffer.n_episodes, val_ratio, seed)
        self.train_mask = ~self.val_mask
        self.horizon, self.pad_before, self.pad_after = horizon, pad_before, pad_after
        self.sampler = self._sampler(self.train_mask)

    def _sampler(self, mask):
        return SequenceSampler(self.replay_buffer, self.horizon, self.pad_before, self.pad_after, episode_mask=mask)

    def get_validation_dataset(self):
        result = copy.copy(self)
        result.sampler = self._sampler(self.val_mask)
        return result

    def get_normalizer(self, **kwargs):
        ends = self.replay_buffer.episode_ends
        starts = np.r_[0, ends[:-1]]
        indices = np.concatenate([np.arange(s,e) for s,e,keep in zip(starts, ends, self.train_mask) if keep])
        normalizer = LinearNormalizer()
        normalizer.fit(dict(action=self.replay_buffer['action'][indices],
                            agent_pos=self.replay_buffer['state'][indices],
                            point_cloud=self.replay_buffer['point_cloud'][indices]), last_n_dims=1, **kwargs)
        return normalizer

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, index):
        sample = self.sampler.sample_sequence(index)
        return dict_apply(dict(obs=dict(point_cloud=sample['point_cloud'].astype(np.float32),
                                        agent_pos=sample['state'].astype(np.float32)),
                               action=sample['action'].astype(np.float32)), torch.from_numpy)
