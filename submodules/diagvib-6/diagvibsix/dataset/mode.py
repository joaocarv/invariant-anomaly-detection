import copy
import random

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

from diagvibsix.dataset.config import *


class Mode(dict):
    """Mode class to provide a structured description of a dataset.
    """
    def __init__(self, mode=None):
        super(Mode, self).__init__()
        # Initialize Mode items.
        if mode is not None:
            self.set_from_dict(mode)

    def get_dict(self):
        """Return only the dictionary.
        """
        mode = dict()
        for key, val in self.items():
            mode[key] = val
        return mode

    def set_from_dict(self, mode):
        """Set Mode items from dictionary.
        """
        for key, val in mode.items():
            self[key] = copy.deepcopy(val)

    def random(self, t, objs):
        """Generate The random Mode with one object.

        :param t: str
            The dataset type, i.e., 'train', 'val', or 'test'.
        :param objs: int
            Number of random objects of this random Mode.
        """
        # Set empty tag.
        self['tag'] = ''
        # Set random objects.
        self['objs'] = [{
            'category': t,
            'shape': list(range(10)),
            'hue': list(HUES),
            'texture': list(TEXTURES),
            'lightness': list(LIGHTNESS),
            'position': list(POSITION),
            'scale': list(SCALE),
        } for _ in range(objs)]
