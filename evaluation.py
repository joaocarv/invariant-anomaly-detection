from omegaconf import OmegaConf
from argparse import ArgumentParser
from copy import deepcopy
from train import preprocess_config
from test import test_impl


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to a evaluation config file")
    args = parser.parse_args()
    return args


def evaluation():
    """Test an anomaly classification and segmentation model that is initially trained via `tools/train.py`.

    The script is able to write the results into both filesystem and a logger such as Tensorboard.
    """
    args = get_args()
    evaluation_cfg = OmegaConf.load(args.config)
    ckpts = evaluation_cfg['ckpt_paths']
    test_cfgs_paths = evaluation_cfg['config_paths']
    l = len(ckpts)
    assert l == len(test_cfgs_paths), "The length of checkpoints and length of cfgs do not match!"
    test_cfgs = []

    for i in range(l):
        test_cfg_path = test_cfgs_paths[i]
        ckpt_path = ckpts[i]
        # rewrite dataset configs
        test_cfg_template = OmegaConf.load(test_cfg_path)
        if evaluation_cfg.project:
            test_cfg_template.project.path = evaluation_cfg.project.path  # change output folder
        if evaluation_cfg.visualization:
            for key,value in evaluation_cfg.visualization.items():
                test_cfg_template.visualization[key] = value
        for dataset_cfg in evaluation_cfg['datasets']:
            test_cfg = deepcopy(test_cfg_template)
            for key, value in dataset_cfg.items():
                test_cfg['dataset'][key] = value
            test_cfg = preprocess_config(test_cfg)
            test_cfg.model.init_weights = ckpt_path
            test_cfgs.append(test_cfg)

    print("-------------------------")
    print("Test cfgs finished loading")
    print("-------------------------")

    for test_cfg in test_cfgs:
        test_impl(test_cfg)


if __name__ == "__main__":
    evaluation()
