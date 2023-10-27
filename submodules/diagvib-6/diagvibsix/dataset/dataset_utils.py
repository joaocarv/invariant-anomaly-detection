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

import numpy as np
import colorsys

from diagvibsix.dataset.config import *

__all__ = ['sample_attribute',
           'get_mt_labels']

# Draw a random factor-class instance from a given factor-class label.
# =============================================================================


def get_position(semantic_attr):
    return np.random.uniform(*POSITION[semantic_attr][0]), np.random.uniform(*POSITION[semantic_attr][1])


def get_scale(semantic_attr):
    return np.random.uniform(*SCALE[semantic_attr])


def get_colorgrad(hue_attr, light_attr):
    l1 = np.random.uniform(*LIGHTNESS[light_attr][0])
    l2 = np.random.uniform(*LIGHTNESS[light_attr][1])
    if hue_attr == 'gray':
        col1, col2 = (0., l1, 0.), (0., l2, 0.)
        col1, col2 = colorsys.hls_to_rgb(*col1), colorsys.hls_to_rgb(*col2)
        return tuple((int(x*255.) for x in col1)), tuple((int(x*255.) for x in col2))
    hue = np.random.uniform(*HUES[hue_attr])
    if hue < 1.:
        hue += 360.
    col1, col2 = (hue / 360., l1, 1.0), (hue / 360., l2, 1.0)
    col1, col2 = colorsys.hls_to_rgb(*col1), colorsys.hls_to_rgb(*col2)
    return tuple((int(x * 255.) for x in col1)), tuple((int(x * 255.) for x in col2))


def sample_attribute(name, semantic_attr, **kwargs):
    get_fn = globals()['get_' + name]
    return get_fn(semantic_attr, **kwargs)


def get_mt_labels(task_label):
    return np.argmax([cls == task_label[1] for cls in OBJECT_ATTRIBUTES[task_label[0]]])
