from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from argparse import ArgumentParser

from anomalib.utils.loggers import get_experiment_logger
from train import preprocess_config, load_data_module, load_model, _get_callbacks
from anomalib.utils.callbacks import get_callbacks
import wandb


def get_args():
    """Get CLI arguments.

    Returns:
        Namespace: CLI arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to a model config file")
    parser.add_argument("--dataset", type=str, required=False, help="Path to the dataset config file")
    parser.add_argument("--weight", type=str, default="Path to ckpt ")
    args = parser.parse_args()
    return args


def test():
    args = get_args()
    config = OmegaConf.load(args.model)
    config.dataset = OmegaConf.load(args.dataset)
    config.model.init_weights = args.weight
    config = preprocess_config(config)
    test_impl(config)


def test_impl(test_cfg):
    datamodule = load_data_module(test_cfg, test_cfg.dataset.format.lower())
    experiment_logger = get_experiment_logger(test_cfg)
    model = load_model(test_cfg)
    callbacks = _get_callbacks(test_cfg)
    trainer = Trainer(callbacks=callbacks, **test_cfg.trainer, logger=experiment_logger)
    print("-------------------------")
    print("----------------Start Testing--------------------")
    print("----------------" + test_cfg.model.name + "-----------------------")
    print("----------------" + test_cfg.dataset.name + "-----------------------")
    print("-------------------------")
    trainer.test(model=model, datamodule=datamodule)
    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    test()
