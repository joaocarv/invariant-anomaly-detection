import os
import copy
import numpy as np

from diagvibsix.auxiliaries import load_obj, save_obj
from diagvibsix.dataset.paint_images import Painter
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

from diagvibsix.dataset.dataset_utils import sample_attribute
from diagvibsix.dataset.config import DATASETS


def random_choice(attr):
    """Returns a random choice of a list of attributes or the single attribute that was provided.

    Args:
        attr (list, str): Single attribute or list of attributes.

    Returns:
        str: Single attribute.

    """

    if isinstance(attr, list):
        return np.random.choice(attr)
    else:
        return attr


def get_answer(semantic_image_spec, question):
    """ Returns the attribute for a certain attribute type of an object.
    Attribute types are e.g. 'category', 'class', 'style'.
    """
    # We never have more than one object and questions can't be ambiguous (otherwise the image couldn't be generated in
    # the first place
    if question == 'category':
        return semantic_image_spec['objs'][0][question].split()[0]
    else:
        return semantic_image_spec['objs'][0][question]


class Dataset(object):
    """Class to provide a structured definition of a dataset.
    We use Modes to define a dataset. For

    """

    def __init__(self, dataset_spec, seed, cache_path=None):
        self.questions_answers = None
        np.random.seed(seed)

        # The specification of the dataset.
        self.spec = copy.deepcopy(dataset_spec)

        # The task, i.e., the predicted factor.
        self.task = dataset_spec['task']

        # Setup painter.
        self.painter = Painter()

        if (cache_path is not None) and (os.path.exists(cache_path)):
            # load cache
            cache_data = load_obj(cache_path)
            self.images = cache_data['images']
            self.image_specs = cache_data['image_specs']
            self.task_labels = cache_data['tasks_labels']
            self.permutation = cache_data['permutation']
            print('Loaded dataset from cache file {}'.format(cache_path))
        else:
            # List of image specification / question / answer of this dataset.
            # Holds the specification dict for each sample.
            self.image_specs = []
            self.images = []
            self.task_labels = []

            # Loop over modes.
            for mode_cntr, mode in enumerate(self.spec['modes']):
                mode['samples'] = int(mode['ratio'] * self.spec['samples'])
                image_specs, images, task_labels = self.draw_mode(mode['specification'], mode['samples'])
                self.image_specs += image_specs
                self.images += images
                self.task_labels += task_labels

            # Permutation of the whole dataset.
            self.permutation = list(range(len(self.images)))
            np.random.shuffle(self.permutation)

            # Save to cache.
            if cache_path is not None:
                cache_data = {
                    'images': self.images,
                    'image_specs': self.image_specs,
                    'task_labels': self.task_labels,
                    'permutation': self.permutation
                }

                # Save cache.
                print('Saved dataset to cache file {}'.format(cache_path))
                save_obj(cache_data, cache_path)

    def draw_mode(self, mode_spec, number_of_samples):
        """Draws the entire mode.
        """
        image_specs = [None for _ in range(number_of_samples)]
        images = [None for _ in range(number_of_samples)]
        task_labels = [None for _ in range(number_of_samples)]

        # Loop over all samples to be added.
        for sample_cntr in range(number_of_samples):
            image_spec, semantic_image_spec = self.draw_image_spec_from_mode(copy.deepcopy(mode_spec))

            # Get answers to all questions
            task_labels[sample_cntr] = get_answer(semantic_image_spec, self.task)

            image_specs[sample_cntr] = image_spec
            image = self.painter.paint_images(image_spec, self.spec['shape'])
            images[sample_cntr] = image

        return image_specs, images, task_labels

    def draw_image_spec_from_mode(self, mode_spec):
        """ Draws a single image specification from a mode.
        """
        # Set empty dictionary for each sample.
        image_spec = dict()

        # Add tag to image specification
        image_spec['tag'] = mode_spec['tag'] if 'tag' in mode_spec.keys() else ''

        # Each attribute in a mode can be given as a list (e.g. 'color': ['red', 'blue', 'green']).
        # In such cases we want to sample an attribute specification randomly from that list.
        # If only a single attribute is given, we use that.
        for attr in (set(mode_spec.keys()) - {'objs', 'tag'}):
            mode_spec[attr] = random_choice(mode_spec[attr])

        # Loop over objects.
        image_spec['objs'] = []
        for obj_spec in mode_spec['objs']:
            # In case list is given for an attribute, sample an attribute specification randomly from that list
            for attr in obj_spec.keys():
                obj_spec[attr] = random_choice(obj_spec[attr])

            obj = dict()

            # Object category / class.
            obj['category'] = obj_spec['category']
            obj['shape'] = obj_spec['shape']
            obj['texture'] = obj_spec['texture']

            # Draw class instance.
            last_instance_idx = DATASETS[obj['category']]['samples'][obj['shape']]
            obj['instance'] = np.random.randint(0, last_instance_idx)
            # Draw object color (hue + lightness).
            obj['color'] = sample_attribute('colorgrad',
                                            obj_spec['hue'],
                                            light_attr=obj_spec['lightness'])
            # Object position / scale.
            for attr in ['position', 'scale']:
                obj[attr] = sample_attribute(attr, obj_spec[attr])

            # Add object to sample.
            image_spec['objs'].append(obj)

        return image_spec, mode_spec

    def getitem(self, idx):
        permuted_idx = self.permutation[idx]

        return {'image': self.images[permuted_idx],
                'targets': (self.task, self.task_labels[permuted_idx]),
                'tag': self.image_specs[permuted_idx]['tag']}
