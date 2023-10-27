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
from diagvibsix.dataset.config import SHARED_STUDY_PATH, FACTORS, FACTOR_CLASSES, IMG_SIZE, EXPERIMENT_SAMPLES, SELECTED_CLASSES_PATH

# Get factors and number of factors.
F = len(FACTOR_CLASSES)

# Define list of all studies.
# Each study is defined by its two parameters: [correlated factors, predicted factors]
# An optional third parameter in [0, 100] might be given to specify a correlation frequency. By default this is 100.
# Study for the biased experiment.
STUDIES = [[0, 1],          # ZSO
           [2, 1, 80],      # FGO-80
           [2, 1, 90],      # FGO-90
           [2, 1, 95],      # FGO-95
           [2, 1, 100]]     # ZGO
"""
    The general folder for a dataset specification is :
        SHARED_DATASET_PATH / study name / factor combination, corr weight / sample id / train,val,test.yml

    Test set:
        In all cases, the test includes:
            * all induced class modes from the training set
                TAG: 'ic'
            * one mode uniform over ALL classes of factors that do NOT get predicted.
                TAG: 'uniform all'
        In case of correlated factors (2 ... F):
            * one mode uniform over only the selected classes (of correlated factors).
              This may contain all possible correlation violations, but in an unstructured way.
                TAG: 'uniform selected'
            * For each correlated factor The three modes that violate this single factor.
                TAG: 'violate FACTOR'
"""


def generate_dataset(corr_comb, pred_comb, corr_weight, selected_classes, random_seed):
    # Fix random seed for re-producebility.
    np.random.seed(random_seed)
    random.seed(random_seed)
    # Get number of classes.
    classes = 3
    # Correlation weight -> correlation ratio.
    corr_ratio = float(corr_weight) / 100.0
    # Mininum number of samples for each induced-class.
    # The first factor indicates the number image samples for the study: F-F
    # Here, classes ** F = 3 ** 6 = 729
    test_sample = {
        'train': 10000,
        'violate correlated': 10000,
        'uniform selected': 10000,
        'uniform all': 10000,
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
        # Generate a mode for every class (= 3 here), already reflecting all correlated factors.
        cls_mode = []
        for cls in range(classes):
            cls_mode.append(Mode())
            cls_mode[-1].random(t, 1)
            # Some adjustments.
            cls_mode[-1]['objs'][0]['category'] = t
            # Set all factors, either to specific class (if correlated) or to free.
            for f in FACTOR_CLASSES:
                if f[0] in corr_comb:
                    # In case this factor is correlated, set to specific factor class.
                    cls_mode[-1]['objs'][0][f[0]] = [selected_classes[f[0]][cls]]
                else:
                    # In case this factor is free, set to full list of factor classes.
                    cls_mode[-1]['objs'][0][f[0]] = selected_classes[f[0]]
        # Collapse modes in case of no factor correlations.
        if corr_comb == ():
            cls_mode = [cls_mode[0]]
        # Determine predicted and free (= not correlated) factors.
        # These generate the induced classes.
        # This might be empty or include all factors.
        pred_not_corr = []
        ic_comb = [[]]
        for f in pred_comb:
            if f not in corr_comb:
                pred_not_corr.append(f)
                new_ic_comb = []
                # Recombine already considered pred_not_corr factors with classes of this new pred_not_corr factor.
                for ic in ic_comb:
                    for cls in selected_classes[f]:
                        new_ic_comb.append(ic + [cls])
                ic_comb = copy.deepcopy(new_ic_comb)
        # Consider pred_not_corr factors.
        # Each combinations of factor classes of the factors in pred_not_corr now induces a class in each cls_mode.
        ic_mode = []    # This will hold one mode for each induced class.
        for cm in cls_mode:
            # Generate a mode for each induced class for this cls_mode.
            for ic in ic_comb:
                # Start with the original cls_mode.
                this_mode = copy.deepcopy(cm.get_dict())
                # Set all pred_not_corr factors to one class combination (= induced class).
                for f_id, f in enumerate(pred_not_corr):
                    this_mode['objs'][0][f] = ic[f_id]
                # Append induced class mode.
                ic_mode.append(this_mode)
        # Set mode ratios to uniform (e.g. uniform induced class probability).
        # Note: corr_ratio should only be != 1.0 in case of 2 correlated factors, 1 predicted, i.e., study 2-1-F.
        if t in ['train', 'val']:
            ratio = corr_ratio / float(len(ic_mode))
        else:
            ratio = (float(test_sample['train']) / float(test_samples)) / float(len(ic_mode))
        # Finally put modes into dataset.
        for ic in ic_mode:
            ic['tag'] = 'ic'
            ds_spec[t]['modes'].append({'specification': copy.deepcopy(ic),
                                        'ratio': ratio})

        # In case of a weighted correlation (not 100 % bias), add additional confounder modes.
        if corr_weight != 100 and len(corr_comb) == 2 and len(pred_comb) == 1 and t in ['train', 'val']:
            # Add one uniform confounder mode per predicted class.
            for cls, cm in enumerate(cls_mode):
                # Start with the original cls_mode.
                confounder_mode = copy.deepcopy(cm.get_dict())
                # Get the correlated factor and factor class of the non-predicted factor.
                for f in FACTOR_CLASSES:
                    # This should be True for only one f (the non-predicted, correlated factor).
                    if f[0] in list(corr_comb) and f[0] not in list(pred_comb):
                        # Get all selected classes of the second factor (bias factor, not predicted).
                        factor_classes_of_corr_factor = copy.deepcopy(selected_classes[f[0]])
                        # Remove the correlated factor class.
                        factor_classes_of_corr_factor.remove(selected_classes[f[0]][cls])
                        # Update confounder mode.
                        confounder_mode['objs'][0][f[0]] = factor_classes_of_corr_factor
                # Set ratio and add mode.
                ratio = (1.0 - corr_ratio) / float(len(ic_mode))
                ds_spec[t]['modes'].append({'specification': copy.deepcopy(confounder_mode), 'ratio': ratio})

        # Add additional test settings.
        if t == 'test':
            # Add one test mode uniform over ALL classes (i.e. never seen classes possible), for all non-pred factors.
            alluni_mode = copy.deepcopy(cls_mode[0].get_dict())
            alluni_mode['tag'] = 'uniform all'
            for f in FACTOR_CLASSES:
                if f[0] in pred_comb:
                    alluni_mode['objs'][0][f[0]] = copy.copy(selected_classes[f[0]])
                else:
                    alluni_mode['objs'][0][f[0]] = f[1]
            # Set mode ratio and add mode.
            ratio = float(test_sample['uniform all']) / float(test_samples)
            ds_spec['test']['modes'].append({'specification': alluni_mode,
                                             'ratio': ratio, })

            if len(corr_comb) > 0:
                # Add one test mode uniform over selected classes (i.e. all violations possible).
                uni_mode = copy.deepcopy(cls_mode[0].get_dict())
                uni_mode['tag'] = 'uniform selected'
                for f in FACTOR_CLASSES:
                    uni_mode['objs'][0][f[0]] = copy.copy(selected_classes[f[0]])
                # Set mode ratio and add mode.
                ratio = float(test_sample['uniform selected']) / float(test_samples)
                ds_spec['test']['modes'].append({'specification': uni_mode,
                                                 'ratio': ratio, })

                # Add three correlation violating modes for each correlated factor.
                viol_factor_mode = []
                for f in FACTOR_CLASSES:
                    # If factor is a correlated one.
                    if f[0] in corr_comb:
                        # Violate this factor for each class mode.
                        for cm in cls_mode:
                            # Start with class mode.
                            this_mode = copy.deepcopy(cm.get_dict())
                            # Get correlated class for this factor.
                            this_factor_class = this_mode['objs'][0][f[0]][0]
                            # Set this one factor to correlation violating classes.
                            viol_classes = copy.copy(selected_classes[f[0]])
                            viol_classes.remove(this_factor_class)
                            this_mode['objs'][0][f[0]] = viol_classes
                            this_mode['tag'] = 'violate ' + f[0]
                            viol_factor_mode.append(this_mode)
                # Set mode ratio and add modes.
                ratio = (float(test_sample['violate correlated']) / float(test_samples)) / float(len(viol_factor_mode))
                # Put modes into test dataset.
                for mode in viol_factor_mode:
                    ds_spec['test']['modes'].append({'specification': copy.deepcopy(mode),
                                                     'ratio': ratio, })

    return ds_spec


def main():
    # Load shared selected classes.
    selected_classes = load_yaml(SELECTED_CLASSES_PATH)

    # Loop over all studies.
    for s_id, study in enumerate(STUDIES):
        # Get study parameters.
        corr_factors = study[0]
        pred_factors = study[1]
        # Get correlation weighting. If none is provided set it to 100
        corr_weight = study[2] if len(study) == 3 else 100
        # Set study name.
        if corr_factors == 0:
            study_name = 'study_ZSO'
        elif corr_factors == 2 and corr_weight == 100:
            study_name = 'study_ZGO'
        else:
            study_name = 'study_FGO-' + str(corr_weight)
        if True:
            print("Generate " + study_name)
        # Generate config folder if not already existing
        study_folder = SHARED_STUDY_PATH + os.sep + study_name
        if not os.path.exists(study_folder):
            os.makedirs(study_folder)
        # Generate factor pairings for correlated factors.
#        corr_factor_combinations = []
#        if corr_factors > 0:
        corr_factor_combinations = list(itertools.combinations(FACTORS, corr_factors))
        # Generate factor pairings for predicted factors.
        pred_factor_combinations = []
        if pred_factors > 0:
            pred_factor_combinations = [list(item) for item in itertools.combinations(FACTORS, pred_factors)]
        # Loop over all correlation combinations.
        for corr_comb in corr_factor_combinations:
            # Loop over all prediction combinations.
            for pred_comb in pred_factor_combinations:
                # Check for [2,1] exception and exclude certain cases.
                if corr_factors == 2 and pred_factors == 1:
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
                    seed = 1332 + samp + s_id * EXPERIMENT_SAMPLES
                    dataset = generate_dataset(corr_comb,
                                               pred_comb,
                                               corr_weight,
                                               selected_classes[samp],
                                               random_seed=seed)
                    # Save experiment (train, val, test) to target folder.
                    save_experiment(dataset, sample_folder)


if __name__ == '__main__':
    main()
