from src.utils.generate_ds_utils import *
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--root", required=True, type=str, help="root_dir")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    generate_env_and_labels(args.root)
