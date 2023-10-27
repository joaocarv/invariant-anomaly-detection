#!/usr/local/bin/python3
# Copyright (c) 2021 Robert Bosch GmbH Copyright holder of the paper "DiagViB-6: A Diagnostic Benchmark Suite for Vision Models in the Presence of Shortcut and Generalization Opportunities" accepted at ICCV 2021.
# All rights reserved.
###
# The paper "DiagViB-6: A Diagnostic Benchmark Suite for Vision Models in the Presence of Shortcut and Generalization Opportunities" accepted at ICCV 2021.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Author: Elias Eulig, Volker Fischer
# -*- coding: utf-8 -*-

import re
import torch
from torch.utils.data import Dataset as TorchDataset
import numpy as np
from diagvibsix.auxiliaries import get_dataset_tags, load_yaml
from diagvibsix.dataset.dataset import Dataset
from diagvibsix.dataset.dataset_utils import get_mt_labels

__all__ = ['TorchDatasetWrapper']


def get_per_ch_mean_std(images):
    """ Images must be a list of (T, 3, W, H) numpy arrays """
    x = np.stack(images, axis=0)
    mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
    std = np.sqrt(((x - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True))
    return torch.from_numpy(mean.squeeze(0)).type(torch.float), torch.from_numpy(std.squeeze(0)).type(torch.float)


class TorchDatasetWrapper(TorchDataset):
    def __init__(self, dataset_spec_path, seed, normalization='z-score', mean=None, std=None, cache=False):
        self.dataset_spec = load_yaml(dataset_spec_path)
        cache_file = '{}.pkl'.format(re.split('.yml|.yaml', dataset_spec_path)[0]) if cache else None
        self.dataset = Dataset(self.dataset_spec, seed, cache_path=cache_file)
        self.normalization = normalization

        # Get tags, task and shape.
        self.tags = get_dataset_tags(self.dataset_spec)
        self.task = self.dataset_spec['task']
        self.shape = self.dataset_spec['shape']

        # Get normalization stats
        self.mean, self.std = mean, std
        self.min = 0.
        self.max = 255.
        if self.normalization == 'z-score' and (self.mean is None or self.std is None):
            self.mean, self.std = get_per_ch_mean_std(self.dataset.images)

    def denormalize(self, X):
        if self.normalization == 'z-score':
            return torch.clamp(X.mul(self.std).add(self.mean), min=0, max=255).type(torch.uint8)
        else:
            return torch.clamp(X.mul(self.max - self.min).add(self.min), min=0, max=255).type(torch.uint8)

    def _normalize(self, X):
        if self.normalization == 'z-score':
            return X.sub_(self.mean).div_(self.std)
        else:
            return X.sub_(self.min).div_(self.max - self.min)

    def _to_T(self, x, dtype):
        return torch.from_numpy(x).type(dtype)

    def __len__(self):
        return len(self.dataset.permutation)

    def __getitem__(self, item):
        sample = self.dataset.getitem(item)
        image, target, tag = sample.values()
        image = self._normalize(self._to_T(image, torch.float))
        target = torch.tensor(get_mt_labels(target), dtype=torch.long)
        return {'image': image, 'target': target, 'tag': tag}
