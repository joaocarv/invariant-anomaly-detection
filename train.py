import logging
import warnings
import random
import os
from importlib import import_module

import numpy as np
import torch
from argparse import ArgumentParser, Namespace

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from anomalib.utils.callbacks import LoadModelCallback, get_callbacks
from anomalib.utils.loggers import configure_logger, get_experiment_logger
from anomalib.data.utils import TestSplitMode
from pathlib import Path
from omegaconf import OmegaConf
from src.data import FolderDg, WildsAnomalyLightning
from wilds.common.grouper import CombinatorialGrouper
from torchmetrics.classification import *
import wilds
import torchmetrics
import pytorch_lightning as pl

logger = logging.getLogger("ciia")
model_dg_list = ["stfpm_mmd", "cflow_mmd", "reverse_distillation_mmd", "cfa_mmd"]


def _min_max_normalize(preds):
    min_p = preds.min()
    max_p = preds.max()
    preds = (preds - min_p) / (max_p - min_p)
    return max_p, min_p, preds


class MetricsCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.metrics = ["precision", "recall", "accuracy", "auroc", "f1"]
        self.metric_class = {"precision": BinaryPrecision, "recall": BinaryRecall,
                             "accuracy": BinaryAccuracy, "auroc": BinaryAUROC, "f1": BinaryF1Score}
        self.metrics_dict = {m: None for m in self.metrics}
        self.metrics_dict["auroc"] = BinaryAUROC(thresholds=1000)
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.threshold = 0.5
        self.labels = []
        self.preds = []

    def on_validation_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs,
            batch,
            batch_idx: int,
            dataloader_idx: int,
    ) -> None:
        labels = outputs["label"]
        preds = outputs["pred_scores"]
        self.labels.append(labels)
        self.preds.append(preds)

    def on_test_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs,
            batch,
            batch_idx: int,
            dataloader_idx: int,
    ) -> None:
        self.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._update_metrics(pl_module)
        self._log_metrics("val")

    def _update_metrics(self, pl_module):
        preds = torch.concat(self.preds).squeeze()
        labels = torch.concat(self.labels).squeeze()
        min_p, max_p = pl_module.normalization_metrics.compute()
        min_p = min_p.to(preds.device)
        max_p = max_p.to(preds.device)
        preds = (preds - min_p) / (max_p - min_p)
        self.threshold = float((pl_module.image_threshold.value - min_p) / (max_p - min_p))
        # max_p, min_p, preds = _min_max_normalize(preds)
        # self.threshold = float((self.threshold - min_p) / (max_p - min_p))
        self.log("image_threshold", self.threshold)

        for metric in self.metrics:
            if metric != "auroc":
                self.metrics_dict[metric] = self.metric_class[metric](threshold=self.threshold)
            self.metrics_dict[metric].update(preds=preds, target=labels)
        abnormals = preds >= self.threshold
        for (i, label) in enumerate(labels):
            if label == 1:
                if abnormals[i]:
                    self.tp += 1
                else:
                    self.fn += 1
            else:
                if abnormals[i]:
                    self.fp += 1
                else:
                    self.tn += 1

    def on_test_epoch_end(self, trainer, pl_module):
        self._update_metrics(pl_module)
        self._log_metrics("test")

    def _log_metrics(self, mode):
        for metric in self.metrics:
            val = self.metrics_dict[metric].compute()
            self.log(f"{mode}/{metric}", val)
            if metric != "auroc":
                self.metrics_dict[metric] = None
        self.log(f"{mode}/tp", self.tp)
        self.log(f"{mode}/fp", self.fp)
        self.log(f"{mode}/tn", self.tn)
        self.log(f"{mode}/fn", self.fn)
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self.preds = []
        self.labels = []
        self.metrics_dict["auroc"].reset()


def _snake_to_pascal_case(model_name: str) -> str:
    """Convert model name from snake case to Pascal case.

    Args:
        model_name (str): Model name in snake case.

    Returns:
        str: Model name in Pascal case.
    """
    return "".join([split.capitalize() for split in model_name.split("_")])


def seed_everything_at_once(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args() -> Namespace:
    """Get command line arguments.

    Returns:
        Namespace: List of arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model config")
    parser.add_argument("--dataset", type=str, required=True, help="Path to a dataset config")
    parser.add_argument("--log-level", type=str, default="INFO", help="<DEBUG, INFO, WARNING, ERROR>")

    args = parser.parse_args()
    return args


def preprocess_config(config):
    project_path = Path(config.project.path) / config.model.name / config.dataset.name
    (project_path / "weights").mkdir(parents=True, exist_ok=True)
    (project_path / "images").mkdir(parents=True, exist_ok=True)
    config.project.path = str(project_path)
    config.dataset["transform_config_train"] = config.dataset.transform_config.train
    config.dataset["transform_config_eval"] = config.dataset.transform_config.eval
    return config


def _get_callbacks(config):
    callbacks = get_callbacks(config)
    callbacks.append(MetricsCallback())
    callbacks.append(ModelCheckpoint(save_last=True, filename="{epoch}-{step}"))
    return callbacks


def train():
    """Train an anomaly classification or segmentation model based on a provided configuration file."""
    args = get_args()
    configure_logger(level=args.log_level)

    if args.log_level == "ERROR":
        warnings.filterwarnings("ignore")

    config = OmegaConf.load(args.model)
    config.dataset = OmegaConf.load(args.dataset)
    train_and_test(config)


def train_and_test(config):
    config = preprocess_config(config)
    if config.project.get("seed") is not None:
        seed_everything_at_once(config.project.seed)
    dataset_format = config.dataset.format.lower()
    datamodule = load_data_module(config, dataset_format)
    model = load_model(config)
    experiment_logger = get_experiment_logger(config)
    callbacks = _get_callbacks(config)
    trainer = Trainer(**config.trainer, logger=experiment_logger, callbacks=callbacks)
    logger.info("Training the model.")
    trainer.fit(model=model, datamodule=datamodule)
    logger.info("Loading the best model weights.")
    load_model_callback = LoadModelCallback(weights_path=trainer.checkpoint_callback.best_model_path)
    trainer.callbacks.insert(0, load_model_callback)  # pylint: disable=no-member
    if config.dataset.test_split_mode == TestSplitMode.NONE:
        logger.info("No test set provided. Skipping test stage.")
    else:
        logger.info("Testing the model.")
        trainer.test(model=model, datamodule=datamodule)
    return trainer, model, datamodule


def load_model(config):
    model_name = config.model.name
    camel_model_name = _snake_to_pascal_case(model_name)
    if model_name in model_dg_list:
        module = import_module("src.model")
    else:
        module = import_module(f"anomalib.models.{model_name}")
    model = getattr(module, f"{camel_model_name}Lightning")(config)
    if "init_weights" in config.model.keys() and config.model.init_weights:
        model.load_state_dict(torch.load(config.model.init_weights)["state_dict"], strict=False)
    return model


def load_data_module(config, dataset_format):
    if dataset_format == "folderdg":
        init_params = {key: config.dataset[key] if key in config.dataset else None for key in FolderDg.get_init_args()}
        datamodule = FolderDg(**init_params)
    elif dataset_format == "wilds":
        wilds_dataset = wilds.get_dataset(dataset=config.dataset.name, download=True)
        grouper = CombinatorialGrouper(wilds_dataset, config.dataset.grouper.groupby_fields)
        params = config.dataset
        if "subsample_ratio" in params:
            subsample_ratio = params.subsample_ratio
        else:
            subsample_ratio = {"train": 1.0, "val": 1.0, "test": 1.0}
        datamodule = WildsAnomalyLightning(wilds_dataset=wilds_dataset, grouper=grouper,
                                           train_bach_size=params.train_batch_size,
                                           eval_batch_size=params.eval_batch_size, num_workers=params.num_workers,
                                           normal_ys=params.normal_ys, anomaly_ys=params.anomaly_ys,
                                           transform_config=params.transform_config, subsample_ratio=subsample_ratio,
                                           splits=params.splits)
    else:
        raise NotImplementedError
    return datamodule


if __name__ == "__main__":
    train()
