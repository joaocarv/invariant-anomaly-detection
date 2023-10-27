from diagvibsix.dataset.config import TEXTURES, HUES, POSITION, SCALE, LIGHTNESS, OBJECT_ATTRIBUTES
from argparse import ArgumentParser
from omegaconf import OmegaConf
from copy import deepcopy
import os
import yaml

features = OBJECT_ATTRIBUTES.keys()
textures = set(TEXTURES.keys())
hues = set(HUES.keys())
scales = set(SCALE.keys())
lightness = set(LIGHTNESS.keys())
positions = set(POSITION.keys())
ood_env_template = OmegaConf.load("tmp/od_template.yml")
all_features_with_values = {key: set(value) for key, value in OBJECT_ATTRIBUTES.items()}


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--base_env", required=True, type=str, help="Path to base env config")
    parser.add_argument("--output", required=True, type=str, help="Path to output dir")
    parser.add_argument("--num", default=1000, type=int, help="Number of samples per od env")
    args = parser.parse_args()
    return args


def get_used_features(config):
    used_features = {key: set() for key in features}
    for mode in config.modes:
        specs = mode.specification.objs[0]
        for f in features:
            used_features[f].update(specs[f])
    return used_features


def generate():
    args = get_args()
    output_dir = args.output
    config = OmegaConf.load(args.base_env)
    used_features = get_used_features(config)
    ood_features = {f: all_features_with_values[f] - used_features[f] for f in features}
    for ood_feature, ood_values in ood_features.items():
        output_path = os.path.join(output_dir, f"{ood_feature}_od")
        os.makedirs(output_path, exist_ok=True)
        for od_value in ood_values:
            od_env = deepcopy(ood_env_template)
            for f in features:
                if f != ood_feature:
                    od_env.modes[0].specification.objs[0][f] = list(used_features[f])
                else:
                    od_env.modes[0].specification.objs[0][f] = [od_value]
            output_filename = f"test_{od_value}_od.yml"
            with open(os.path.join(output_path, output_filename), "w+") as f:
                yaml.dump(OmegaConf.to_container(od_env), f, default_flow_style=False)


if __name__ == "__main__":
    generate()
