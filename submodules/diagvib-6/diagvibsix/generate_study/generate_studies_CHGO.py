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
import copy
import numpy as np
import random

from diagvibsix.auxiliaries import save_experiment, load_yaml
from diagvibsix.dataset.mode import Mode
from diagvibsix.dataset.config import SHARED_STUDY_PATH, FACTORS, FACTOR_CLASSES, IMG_SIZE, EXPERIMENT_SAMPLES, SELECTED_CLASSES_PATH

# Get factors and number of factors.
F = len(FACTOR_CLASSES)

"""
    This script generates the study for compositional generalization in a hybrid setting.

    Here one factor class remains fully correlated the other two stay un-correlated.
"""
STUDIES = [0]

"""

    The general folder for a dataset specification is :
        SHARED_DATASET_PATH / study_hybrid / factor combination / sample id / train,val,test.yml

    Test set:
        In all cases, the test includes:
            * all induced class modes from the training set
              This yields three modes. One for the single fully correlated factor class comb and two for the free
              uniform factor combs.
                TAG: 'ic'
            * For the fully correlated factor class, we test its two violated cases. (-> 1 mode)
                TAG: 'violate corr'
            * The free factor classes get predicted. (-> 1 mode)
                TAG: 'violate free'
"""


def generate_dataset(corr_comb, selected_classes, random_seed):
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
        'violate corr': 10000,
        'violate free': 10000,
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
            'task': corr_comb[0],
            'samples': samples[t],
            'shape': [1, IMG_SIZE, IMG_SIZE],
            'correlated factors': list(corr_comb),
        }
        # Set ic ratios uniform, one mode per predicted factor class.
        ic_ratio = 1.0 / float(classes)

        # Add fully correlated single mode.
        fcc_mode = Mode()
        fcc_mode.random(t, 1)
        # Some adjustments.
        fcc_mode['objs'][0]['category'] = t
        # Set factors to free, except for correlated, there use 0-th selected class.
        for f in FACTOR_CLASSES:
            if f[0] in corr_comb:
                fcc_mode['objs'][0][f[0]] = [selected_classes[f[0]][0]]
            else:
                fcc_mode['objs'][0][f[0]] = selected_classes[f[0]]
        fcc_mode['tag'] = 'ic'
        if t in ['train', 'val']:
            ds_spec[t]['modes'].append({'specification': copy.deepcopy(fcc_mode.get_dict()),
                                        'ratio': ic_ratio})
        else:
            ratio = ic_ratio * float(test_sample['train'] / float(test_samples))
            ds_spec[t]['modes'].append({'specification': copy.deepcopy(fcc_mode.get_dict()),
                                        'ratio': ratio})

        # Add the two uniform (free modes).
        for fm in range(2):
            fcc_mode = Mode()
            fcc_mode.random(t, 1)
            # Some adjustments.
            fcc_mode['objs'][0]['category'] = t
            # Set factors to free, except for correlated, there use 0-th selected class.
            for f in FACTOR_CLASSES:
                if f[0] == corr_comb[0]:
                    # This is the predicted factor class of the first factor.
                    fcc_mode['objs'][0][f[0]] = [selected_classes[f[0]][fm + 1]]
                elif f[0] == corr_comb[1]:
                    # This is the actual generalization opportunity over the second factor.
                    fcc_mode['objs'][0][f[0]] = [selected_classes[f[0]][1],
                                                 selected_classes[f[0]][2]]
                else:
                    fcc_mode['objs'][0][f[0]] = selected_classes[f[0]]
            fcc_mode['tag'] = 'ic'
            if t in ['train', 'val']:
                ds_spec[t]['modes'].append({'specification': copy.deepcopy(fcc_mode.get_dict()),
                                            'ratio': ic_ratio})
            else:
                ratio = ic_ratio * float(test_sample['train'] / float(test_samples))
                ds_spec[t]['modes'].append({'specification': copy.deepcopy(fcc_mode.get_dict()),
                                            'ratio': ratio})

        # Add the violating test cases.
        if t == "test":
            fcc_mode = Mode()
            fcc_mode.random(t, 1)
            # Some adjustments.
            fcc_mode['objs'][0]['category'] = t
            for f in FACTOR_CLASSES:
                if f[0] == corr_comb[0]:
                    # This is the predicted factor class of the first factor.
                    fcc_mode['objs'][0][f[0]] = [selected_classes[f[0]][0]]
                elif f[0] == corr_comb[1]:
                    # This is the actual generalization opportunity over the second factor.
                    fcc_mode['objs'][0][f[0]] = [selected_classes[f[0]][1],
                                                 selected_classes[f[0]][2]]
                else:
                    fcc_mode['objs'][0][f[0]] = selected_classes[f[0]]
            fcc_mode['tag'] = 'violate corr'
            ratio = float(test_sample['violate corr'] / float(test_samples))
            ds_spec[t]['modes'].append({'specification': copy.deepcopy(fcc_mode.get_dict()),
                                        'ratio': ratio})

            fcc_mode = Mode()
            fcc_mode.random(t, 1)
            # Some adjustments.
            fcc_mode['objs'][0]['category'] = t
            for f in FACTOR_CLASSES:
                if f[0] == corr_comb[0]:
                    # This is the predicted factor class of the first factor.
                    fcc_mode['objs'][0][f[0]] = [selected_classes[f[0]][1],
                                                 selected_classes[f[0]][2]]
                elif f[0] == corr_comb[1]:
                    # This is the factor class of the non-pred factor.
                    fcc_mode['objs'][0][f[0]] = [selected_classes[f[0]][0]]
                else:
                    fcc_mode['objs'][0][f[0]] = selected_classes[f[0]]
            fcc_mode['tag'] = 'violate free'
            ratio = float(test_sample['violate free'] / float(test_samples))
            ds_spec[t]['modes'].append({'specification': copy.deepcopy(fcc_mode.get_dict()),
                                        'ratio': ratio})

    return ds_spec


def main():
    # Load shared selected classes.
    selected_classes = load_yaml(SELECTED_CLASSES_PATH)

    # Loop over all studies.
    for s_id, study in enumerate(STUDIES):
        # Set study name.
        study_name = 'study_CHGO'
        if True:
            print("Generate " + study_name)
        # Generate config folder if not already existing
        study_folder = SHARED_STUDY_PATH + os.sep + study_name
        if not os.path.exists(study_folder):
            os.makedirs(study_folder)
        # Generate pairings of factor combinations.
        corr_factor_combinations = []
        for f1 in FACTORS:
            for f2 in FACTORS:
                if f1 != f2:
                    corr_factor_combinations.append([f1, f2])
        # Loop over all correlation combinations.
        for corr_comb in corr_factor_combinations:
            # Generate factor naming, incl. corr and pred.
            factor_combination_name = 'HCORR'
            for f in range(len(list(corr_comb))):
                factor_combination_name += '-' + corr_comb[f]
            factor_combination_name += '_PRED-' + corr_comb[0]
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
                seed = 1332 + samp + s_id * EXPERIMENT_SAMPLES
                dataset = generate_dataset(corr_comb, selected_classes[samp], random_seed=seed)
                # Save experiment (train, val, test) to target folder.
                save_experiment(dataset, sample_folder)


if __name__ == '__main__':
    main()