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

# Definition of shared paths.
# =============================================================================
# MNIST save / load file names for pre-processing.
SHARED_LOADPATH_MNIST = './tmp/mnist.npz'
SHARED_SAVEPATH_MNIST = './tmp/mnist_processed.npz'
# Folder to store datasets for studies.
SHARED_STUDY_PATH = './tmp/diagvibsix/studies_selected/'
# Path to the local textures in the repository.
TEXTURE_PATH = 'submodules/diagvib-6/diagvibsix/dataset/textures/'
# Path to the selected factor classes
SELECTED_CLASSES_PATH = 'submodules/diagvib-6/diagvibsix/dataset/selected_classes.yml'
# Path to selected generalization opportunities
SELECTED_GENOPPS_PATH = 'submodules/diagvib-6/diagvibsix/dataset/selected_genops.yml'

# Used image size in px.
IMG_SIZE = 128
# Number of samples per experiment.
EXPERIMENT_SAMPLES = 5
# Definition of the factor-classes for the six factors.
# =============================================================================
# Define texture factor-classes.
TEXTURES = {
    'tiles': os.path.join(TEXTURE_PATH, 'tiles.png'),
    'wood': os.path.join(TEXTURE_PATH, 'wood.png'),
    'carpet': os.path.join(TEXTURE_PATH, 'carpet.png'),
    'bricks': os.path.join(TEXTURE_PATH, 'bricks.png'),
    'lava': os.path.join(TEXTURE_PATH, 'lava.png'),
}
# Define hue factor-classes.
HUES = {
    'red': (0 - 15, 0 + 15),
    'yellow': (60 - 15, 60 + 15),
    'green': (120 - 15, 120 + 15),
    'cyan': (180 - 15, 180 + 15),
    'blue': (240 - 15, 240 + 15),
    'magenta': (300 - 15, 300 + 15),
}
# Define position factor-classes.
POSITION = {
    'upper left': ((1 / 7, 2 / 7), (1 / 7, 2 / 7)),
    'center left': ((1 / 7, 2 / 7), (3 / 7, 4 / 7)),
    'lower left': ((1 / 7, 2 / 7), (5 / 7, 6 / 7)),
    'upper center': ((3 / 7, 4 / 7), (1 / 7, 2 / 7)),
    'center center': ((3 / 7, 4 / 7), (3 / 7, 4 / 7)),
    'lower center': ((3 / 7, 4 / 7), (5 / 7, 6 / 7)),
    'upper right': ((5 / 7, 6 / 7), (1 / 7, 2 / 7)),
    'center right': ((5 / 7, 6 / 7), (3 / 7, 4 / 7)),
    'lower right': ((5 / 7, 6 / 7), (5 / 7, 6 / 7)),
}
# Define scale factor-classes.
SCALE = {
    'small': (1 / 1.45, 1 / 1.35),
    'smaller': (1 / 1.25, 1 / 1.15),
    'normal': (1 / 1.05, 1.05),
    'larger': (1.15, 1.25),
    'large': (1.35, 1.45),
}
# Define lightness factor-classes.
LIGHTNESS = {
    'dark': ((0., 1 / 11), (4 / 11, 5 / 11)),
    'darker': ((2 / 11, 3 / 11), (6 / 11, 7 / 11)),
    'brighter': ((4 / 11, 5 / 11), (8 / 11, 9 / 11)),
    'bright': ((6 / 11, 7 / 11), (10 / 11, 1.)),
}

OBJECT_ATTRIBUTES = {
    'shape': [i for i in range(10)],
    'hue': list(HUES),
    'lightness': list(LIGHTNESS),
    'texture': list(TEXTURES),
    'position': list(POSITION),
    'scale': list(SCALE),
}

# Aggregated definition of all factor-classes.
# =============================================================================
FACTOR_CLASSES = [
    ("position", list(POSITION)),
    ("hue", list(HUES)),
    ("lightness", list(LIGHTNESS)),
    ("scale", list(SCALE)),
    ("shape", list(range(10))),
    ("texture", list(TEXTURES)),
]
FACTORS = [f[0] for f in FACTOR_CLASSES]

# Dataset definition.
# =============================================================================
DATASETS = {
    'train': {
        'classes': [str(i) for i in range(10)],
        'samples': [4738, 5393, 4766, 4904, 4673, 4336, 4734, 5012, 4680, 4759],
        'savepath': SHARED_SAVEPATH_MNIST,
        'size': 40
    },
    'val': {
        'classes': [str(i) for i in range(10)],
        'samples': [1185, 1349, 1192, 1227, 1169, 1085, 1184, 1253, 1171, 1190],
        'savepath': SHARED_SAVEPATH_MNIST,
        'size': 40
    },
    'test': {
        'classes': [str(i) for i in range(10)],
        'samples': [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009],
        'savepath': SHARED_SAVEPATH_MNIST,
        'size': 40
    },
}
