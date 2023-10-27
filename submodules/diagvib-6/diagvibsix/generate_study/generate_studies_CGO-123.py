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
import itertools
import copy
import numpy as np
import random

from diagvibsix.auxiliaries import save_experiment, load_yaml
from diagvibsix.dataset.mode import Mode
from diagvibsix.dataset.config import SHARED_STUDY_PATH, FACTOR_CLASSES, IMG_SIZE, FACTORS, EXPERIMENT_SAMPLES, \
    SELECTED_CLASSES_PATH, SELECTED_GENOPPS_PATH

# Get factors and number of factors.
F = len(FACTOR_CLASSES)

"""
    This script generates the study for compositional generalization successively adding single new
    combinations of factor-classes to the fully correlated study.
"""
STUDIES = [1, 2, 3]

"""

    The general folder for a dataset specification is :
        SHARED_DATASET_PATH / study_compgen_1/2/3 / factor combination / sample id / train,val,test.yml

    Test set:
        In all cases, the test includes:
            * all induced class modes from the training set
                TAG: 'ic'
            * everything else than the ic factor-class combinations.
                TAG: 'violate'
"""


def generate_dataset(study, corr_comb, pred_comb, selected_classes, genopps, random_seed):
    # Fix random seed for re-producebility.
    np.random.seed(random_seed)
    random.seed(random_seed)
    # Get number of classes.
    classes = 3
    # Mininum number of samples for each induced-class.
    # The first factor indicates the number image samples for the study: F-F
    # Here, classes ** F = 3 ** 6 = 729
    test_sample = {
        'train': 10000,
        'violate': 10000,
    }
    test_samples = sum([test_sample[s] for s in test_sample])
    samples = {
        'train': 60 * (classes ** F),
        'val':   12 * (classes ** F),
        'test':  test_samples,
    }
    # Dataset dictionary with items 'train', 'val', 'test'.
    ds_spec = dict()
    for t in ['train', 'val', 'test']:
        # Start with empty template dataset.
        ds_spec[t] = {
            'modes': [],
            'task': pred_comb[0],
            'samples': samples[t],
            'shape': [1, IMG_SIZE, IMG_SIZE],
            'correlated factors': list(corr_comb),
        }
        # Set single mode ratios.
        ic_ratio = 1.0 / float(classes + study)
        viol_ratio = 1.0 / float(classes**2 - classes - study)
        # Add one mode for each combination of selected classes for the two correlated factors.
        for fc1 in range(classes):
            for fc2 in range(classes):
                fcc_mode = Mode()
                fcc_mode.random(t, 1)
                # Some adjustments.
                fcc_mode['objs'][0]['category'] = t
                # Set all factors to free.
                for f in FACTOR_CLASSES:
                    fcc_mode['objs'][0][f[0]] = selected_classes[f[0]]
                # Set correlated factors to this fc1/2 combination.
                # Note that this also makes fc2 the NON-predicted factor (class).
                if pred_comb[0] == corr_comb[0]:
                    fcc_mode['objs'][0][corr_comb[0]] = [selected_classes[corr_comb[0]][fc1]]
                    fcc_mode['objs'][0][corr_comb[1]] = [selected_classes[corr_comb[1]][fc2]]
                else:
                    fcc_mode['objs'][0][corr_comb[0]] = [selected_classes[corr_comb[0]][fc2]]
                    fcc_mode['objs'][0][corr_comb[1]] = [selected_classes[corr_comb[1]][fc1]]
                # Determine if this combination is a training (ic) mode or a testing (no ic) mode.
                fcc_mode['tag'] = 'violate'
                # Add fully correlated factor-class combinations to ic.
                if fc1 == fc2:
                    fcc_mode['tag'] = 'ic'
                # Check if non-predicted, but correlated factor matches chosen fc combination.
                for s in range(study):
                    if fc1 == s and fc2 == genopps[s]:
                        fcc_mode['tag'] = 'ic'

                # Add mode.
                if t in ['train', 'val']:
                    # Only add 'ic' modes.
                    if fcc_mode['tag'] == 'ic':
                        ds_spec[t]['modes'].append({'specification': copy.deepcopy(fcc_mode.get_dict()),
                                                    'ratio': ic_ratio})
                else:
                    # Add ic and violate modes with according test ratios.
                    if fcc_mode['tag'] == 'ic':
                        ratio = ic_ratio * float(test_sample['train'] / float(test_samples))
                        ds_spec[t]['modes'].append({'specification': copy.deepcopy(fcc_mode.get_dict()),
                                                    'ratio': ratio})
                    elif fcc_mode['tag'] == 'violate':
                        ratio = viol_ratio * float(test_sample['violate'] / float(test_samples))
                        ds_spec[t]['modes'].append({'specification': copy.deepcopy(fcc_mode.get_dict()),
                                                    'ratio': ratio})
    return ds_spec


def main():
    selected_classes = load_yaml(SELECTED_CLASSES_PATH)
    genopps = load_yaml(SELECTED_GENOPPS_PATH)

    # Loop over all studies.
    for s_id, study in enumerate(STUDIES):
        # Set study name.
        study_name = 'study_CGO-' + str(study)
        if True:
            print("Generating " + study_name)
        # Generate config folder if not already existing
        study_folder = SHARED_STUDY_PATH + os.sep + study_name
        if not os.path.exists(study_folder):
            os.makedirs(study_folder)
        # Generate pairings of factor combinations.
        corr_factor_combinations = list(itertools.combinations(FACTORS, 2))
        # Generate factor pairings for predicted factors.
        pred_factor_combinations = [[f] for f in FACTORS]
        # Loop over all correlation combinations.
        for corr_comb in corr_factor_combinations:
            # Loop over all prediction combinations.
            for pred_comb in pred_factor_combinations:
                # Check for [2,1] exception and exclude certain cases.
                if pred_comb[0] not in corr_comb:
                    continue
                # Generate factor naming, incl. corr and pred.
                factor_combination_name = 'CORR'
                for f in range(len(list(corr_comb))):
                    factor_combination_name += '-' + corr_comb[f]
                factor_combination_name += '_PRED'
                for f in range(len(list(pred_comb))):
                    factor_combination_name += '-' + pred_comb[f]
                # Generate config folder if not already existing.
                factor_combination_folder = study_folder + os.sep + factor_combination_name
                if not os.path.exists(factor_combination_folder):
                    os.makedirs(factor_combination_folder)
                # Loop over samples.
                for samp in range(EXPERIMENT_SAMPLES):
                    # Generate sample folder.
                    sample_folder = factor_combination_folder + os.sep + str(samp)
                    if not os.path.exists(sample_folder):
                        os.makedirs(sample_folder)
                    # Generate a sample (train, val, test) of this dataset.
                    seed = 1332 + samp
                    dataset = generate_dataset(study,
                                               corr_comb,
                                               pred_comb,
                                               selected_classes[samp],
                                               genopps[samp],
                                               random_seed=seed)
                    # Save experiment (train, val, test) to target folder.
                    save_experiment(dataset, sample_folder)


if __name__ == '__main__':
    main()