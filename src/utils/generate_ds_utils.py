import pickle
from PIL import Image
import os
from copy import deepcopy
import pandas as pd
import re
import csv
import numpy as np
import shutil
from omegaconf import OmegaConf
from time import time
import subprocess
import os
from pathlib import Path
from diagvibsix.dataset import dataset
from diagvibsix.auxiliaries import load_yaml
from diagvibsix.dataset.dataset_utils import get_mt_labels, sample_attribute
from diagvibsix.wrappers import get_per_ch_mean_std
from diagvibsix.dataset.config import DATASETS

OBJECT_ATTRIBUTES = {'shape': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                     'hue': ['red', 'yellow', 'green', 'cyan', 'blue', 'magenta'],
                     'lightness': ['dark', 'darker', 'brighter', 'bright'],
                     'texture': ['tiles', 'wood', 'carpet', 'bricks', 'lava'],
                     'position': ['upper left',
                                  'center left',
                                  'lower left',
                                  'upper center',
                                  'center center',
                                  'lower center',
                                  'upper right',
                                  'center right',
                                  'lower right'],
                     'scale': ['small', 'smaller', 'normal', 'larger', 'large']}

FACTORS = deepcopy(OBJECT_ATTRIBUTES)
FACTORS['position_factor'] = FACTORS.pop('position')
FACTORS['scale_factor'] = FACTORS.pop('scale')

__all__ = ["generate_mead_env", "generate_env_and_labels"]


def _tag2number(metadata):
    metadata_out = [FACTORS[key].index(metadata[key]) for key in FACTORS.keys()]
    return metadata_out


def make_archive(source, destination):
    base = os.path.basename(destination)
    name = base.split('.')[0]
    format = base.split('.')[1]
    archive_from = os.path.dirname(source)
    archive_to = os.path.basename(source.strip(os.sep))
    shutil.make_archive(name, format, archive_from, archive_to)
    shutil.move('%s.%s' % (name, format), destination)


def get_per_ch_mean_std(images):
    """ Images must be a list of (T, 3, W, H) numpy arrays """
    x = np.stack(images, axis=0)
    mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
    std = np.sqrt(((x - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True))
    return mean.squeeze(0).flatten().tolist(), std.squeeze(0).flatten().tolist()


class CreateEnv(dataset.Dataset):

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
            mode_spec[attr] = dataset.random_choice(mode_spec[attr])

        # Loop over objects.
        image_spec['objs'] = []
        for obj_spec in mode_spec['objs']:
            # In case list is given for an attribute, sample an attribute specification randomly from that list
            for attr in obj_spec.keys():
                obj_spec[attr] = dataset.random_choice(obj_spec[attr])

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
            obj['hue'] = obj_spec['hue']
            obj['lightness'] = obj_spec['lightness']
            obj['position_factor'] = obj_spec['position']
            obj['scale_factor'] = obj_spec['scale']

            # Object position / scale.
            for attr in ['position', 'scale']:
                obj[attr] = sample_attribute(attr, obj_spec[attr])

            # Add object to sample.
            image_spec['objs'].append(obj)

        return image_spec, mode_spec


def pickle2dataset(file_path):
    new_dir, exp_name = os.path.split(file_path[:-4])
    new_dir = os.path.join(os.path.join(new_dir, 'dataset'), exp_name)
    os.makedirs(new_dir, exist_ok=True)
    new_dir_images = os.path.join(new_dir, "images")
    os.makedirs(new_dir_images, exist_ok=True)

    file = open(file_path, 'rb')
    data = pickle.load(file)
    file.close()
    mean, std = get_per_ch_mean_std(data['images'])
    with open(os.path.join(new_dir, 'mean_std.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(mean)
        writer.writerow(std)

    specs = [_tag2number(spec['objs'][0]) for spec in data['image_specs']]
    df = pd.DataFrame(data=specs, columns=FACTORS)
    df.insert(0, 'task_labels', data['task_labels'])
    df.insert(7, 'permutation', data['permutation'])
    df.to_csv(os.path.join(new_dir, 'metada.csv'), encoding='utf-8')
    for i, image in enumerate(data['images']):
        image = np.moveaxis(image, 0, 2)
        im = Image.fromarray(image)
        im.save(os.path.join(new_dir_images, str(i) + ".jpeg"))
    print("Unpickled dataset in file " + file_path)


def _find_env_yaml(root_dir, prefix, pkl_files):
    return [os.path.join(root_dir, path) for path in os.listdir(root_dir) if
            path.split('_')[0] == prefix and path.split('.')[0] not in pkl_files and path.endswith('yml')]


def prep_dataset_diagvib_envs(root_dir):
    pkl_files = [file.split('.')[0] for file in os.listdir(root_dir) if file.endswith('pkl')]
    path_train_envs = _find_env_yaml(root_dir, "train", pkl_files)
    path_val_envs = _find_env_yaml(root_dir, "val", pkl_files)
    path_test_envs = _find_env_yaml(root_dir, "test", pkl_files)

    # Generate Images Train
    train_envs_ds = []
    _generate_image(path_train_envs)
    _generate_image(path_val_envs)
    _generate_image(path_test_envs)
    # From pickle to jpeg
    path_pkls = [os.path.join(root_dir, path) for path in os.listdir(root_dir) if
                 path.endswith('pkl')]

    for path in path_pkls:
        pickle2dataset(path)


def _generate_image(path_envs):
    for path_train_env in path_envs:
        print('Generating images: ', path_train_env)
        spec_env = load_yaml(path_train_env)
        env_name = Path(path_train_env).stem
        path_pkl = f"{Path(path_train_env).parent}/{env_name}.pkl"
        # path_pkl = '{}.pkl'.format(re.split('.yml|.yaml', spec_train_env)[0])

        _ = CreateEnv(dataset_spec=spec_env,
                      seed=123,
                      cache_path=path_pkl)


def generate_mead_env(root_dir):
    prep_dataset_diagvib_envs(root_dir=root_dir)


def generate_env_and_labels(root_dir):
    generate_mead_env(root_dir)
    yml_files = [file.split('.')[0] for file in os.listdir(root_dir) if file.endswith('.yml')]
    for filename in yml_files:
        config = OmegaConf.load(os.path.join(root_dir, filename + ".yml"))
        path_to_dataset = os.path.join(root_dir, "dataset", filename)
        labels = []
        label_index = 0
        if os.path.isdir(path_to_dataset):
            all_samples = config.samples
            for mode in config.modes:
                num_samples = int(mode.ratio * all_samples)
                labels += ([label_index] * num_samples)
                label_index += 1
        out_df = pd.DataFrame(labels, columns=['label'])
        out_path = os.path.join(path_to_dataset, "env_label.csv")
        out_df.to_csv(out_path)
