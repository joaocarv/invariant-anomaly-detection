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

import os
import pickle

import yaml

__all__ = ['save_yaml',
           'load_yaml',
           'save_obj',
           'load_obj',
           'get_dataset_tags',
           'save_experiment',
           'get_corr_pred'
           ]


def save_yaml(obj, path):
    """Save yaml object to specified path.
    """
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as outfile:
        yaml.dump(obj, outfile, default_flow_style=False)


def load_yaml(filepath):
    """Load yaml object from specified path.
    """
    return yaml.load(open(filepath), Loader=yaml.FullLoader)


def save_obj(obj, path):
    """Save pickle object to specified path.
    """
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    """Load pickle object from specified path.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_experiment(d_spec, target_folder):
    """Save dataset specifications for experiment to yaml file.
    """
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    for t in d_spec:
        save_yaml(d_spec[t], os.path.join(target_folder, str(t) + '.yml'))


def get_dataset_tags(spec):
    """Given a spec dict return a list of its tags.
    """
    tags = [mode['specification']['tag'] for mode in spec['modes']]
    return list(set(tags))


def get_corr_pred(study_name):
    """
    A study name is of the form CORR-factor1-factor2-factor3_PRED-factor1.
    If no factors are correlated, then CORR_PRED-factor1.
    This function returns lists of correlated and predicted factors from this string.
    """
    if 'CORR-' in study_name:
        # There are correlations
        corrs, preds = study_name.split('CORR-')[1].split('_PRED-')
        corrs = corrs.split('-')
        preds = preds.split('-')
    else:
        # There are no correlations
        corrs = []
        preds = study_name.split('PRED-')[1].split('-')
    return corrs, preds
