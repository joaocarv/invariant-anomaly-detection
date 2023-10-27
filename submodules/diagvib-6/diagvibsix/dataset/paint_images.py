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
from imageio import imread
from skimage.transform import rescale

from diagvibsix.dataset.config import TEXTURES, DATASETS

THRESHOLD = 150

__all__ = ['Painter']


def random_crop(image, size=(128, 128), axes=(-2, -1)):
    """Perform random crop of a given image.

    Parameters
    ----------
    image : ndarray
        Image array from which to crop
    size : tuple
        Tuple specifying the crop size along the two dimensions. Default=(128, 128)
    axes : tuple
        Axes that define the dimension in which to crop. Default=(-2, -1), last two axes.

    Returns
    -------
    ndarray
        Image crop of the given size
    """

    x = np.random.randint(image.shape[axes[0]] - size[0])
    y = np.random.randint(image.shape[axes[1]] - size[1])
    slc = [slice(None)] * image.ndim
    slc[axes[0]] = slice(x, x + size[0])
    slc[axes[1]] = slice(y, y + size[1])
    return image[tuple(slc)]


def rel_to_abs_pos(rel_pos, img_size, obj_size):
    return int((img_size - 1) * rel_pos - (obj_size - 1) / 2)


def pos_to_abs(rel_pos, img_shape, obj_size):
    abs_pos = (rel_to_abs_pos(rel_pos[0], img_shape[1], obj_size[1]),
               rel_to_abs_pos(rel_pos[1], img_shape[2], obj_size[2]))
    return abs_pos


def load_dataset(dataset):
    """For a given dataset dict load the mnist data.

    Parameters
    ----------
    dataset : dict
        DATASET dictionary that contains for each dataset the dataset savepath. Must be a numpy .npz file.

    Returns
    -------
    dict
        dataset dictionary that contains an additional key 'X' for each dataset with the dataset.
    """

    for d_name, d in dataset.items():
        x = np.load(d['savepath'])['x_' + d_name]
        y = np.load(d['savepath'])['y_' + d_name]
        sorted_idxs = y.argsort()
        x = x[sorted_idxs]
        d['X'] = x
    return dataset


class Painter(object):
    """Wrapper class for image generation.

    Note: This avoids the loading of textures for every image that is painted.
    """
    def __init__(self):
        self.textures = {texture: imread(path) for texture, path in TEXTURES.items()}

        self.data_loaded = load_dataset(DATASETS)

    def create_canvas(self, shape):
        """Create and return an empty gray image of the given shape.
        """
        img = np.stack([np.full(shape, 127, dtype='uint16') for _ in range(3)], axis=1)
        return img

    def create_object(self, obj_spec):
        obj_dataset = self.data_loaded[obj_spec['category']]
        obj_plain = obj_dataset['X'][sum(obj_dataset['samples'][:obj_spec['shape']]) + obj_spec['instance']]
        obj_plain = obj_plain.astype('uint16')

        # Get alpha mask to blend with background
        alpha = np.where(obj_plain.copy() > THRESHOLD, 255, 0).astype('uint8')
        alpha = np.stack([alpha] * 3, axis=0)

        #  Create object.
        col = obj_spec['color']
        texture = random_crop(self.textures[obj_spec['texture']], size=obj_plain.shape).astype('uint16')
        obj = np.stack([(col[0][c] * texture + col[1][c] * (255 - texture)) // 255 for c in range(3)], axis=0)

        return obj, alpha, alpha

    def paint_images(self, spec, shape):
        """Paint an image from a given sample specification.

        Parameters
        ----------
        spec : dict
            Sample specification dictionary containing object attributes.
        shape : tuple
            Desired sample size (1, XSize, YSize). Default=(1, 128, 128).
        rescale_factor : float, int
            Rescale factor to use for each of the objects. Default size of each object is 28x28.

        Returns
        -------
        ndarray
            Painted sample of shape 3 x XSize x YSize.
        """

        # Create canvas.
        img = self.create_canvas(shape)

        # Create objects.
        for obj_idx, obj_spec in enumerate(spec['objs']):
            obj, alpha, seg = self.create_object(obj_spec)

            # Place object on canvas
            scale = obj_spec['scale']

            this_obj = rescale(obj.transpose((1, 2, 0)), scale, anti_aliasing=True, preserve_range=True,
                               multichannel=True).transpose(2, 0, 1).astype('uint16')
            this_alpha = rescale(alpha.transpose((1, 2, 0)), scale, anti_aliasing=True, preserve_range=True,
                                 multichannel=True).transpose(2, 0, 1).astype('uint16')

            pos = pos_to_abs(obj_spec['position'], shape, this_obj.shape)

            min_x, min_y = pos[0], pos[1]
            obj_corners = [min_x, min_x + this_obj.shape[1], min_y, min_y + this_obj.shape[2]]
            # Account for cases where the object is (partially) outside the image
            obj_img_corners = [np.max([0, obj_corners[0]]), np.min([img.shape[2], obj_corners[1]]),
                               np.max([0, obj_corners[2]]), np.min([img.shape[3], obj_corners[3]])]
            diff = [np.clip(np.abs(obj_img_corners[i] - obj_corners[i]),
                            a_min=None, a_max=this_obj.shape[axis]) for i, axis in zip(range(4), [1] * 2 + [2] * 2)]

            img_crop = img[0, :, obj_img_corners[0]:obj_img_corners[1], obj_img_corners[2]:obj_img_corners[3]]

            # If this_obj is partially outside img, then crop it accordingly. Do the same for alpha and seg
            this_obj = this_obj[:, diff[0]:this_obj.shape[1] - diff[1], diff[2]:this_obj.shape[2] - diff[3]]
            this_alpha = this_alpha[:, diff[0]:this_alpha.shape[1] - diff[1], diff[2]:this_alpha.shape[2] - diff[3]]

            this_obj = (this_alpha * this_obj + (255 - this_alpha) * img_crop) // 255
            img[0, :, obj_img_corners[0]:obj_img_corners[1], obj_img_corners[2]:obj_img_corners[3]] = this_obj

        return img[0, :, :, :].astype('uint8')
