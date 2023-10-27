"""STFPM: Student-Teacher Feature Pyramid Matching for Unsupervised Anomaly Detection.

https://arxiv.org/abs/2103.04257
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from omegaconf import DictConfig
from anomalib.models.stfpm import StfpmLightning
from src.loss_functions import MMDLoss
from .dg_module import DgLightning
__all__ = ["StfpmMmdLightning"]


class StfpmMmdLightning(StfpmLightning, DgLightning):

    def __init__(
            self,
            hparams: DictConfig
    ):
        StfpmLightning.__init__(self, hparams)
        DgLightning.__init__(self, pooled=True, dataset_format=hparams.dataset.format,
                             dataset_name=hparams.dataset.name, alpha=hparams.model.alpha,
                             sigmas=hparams.model.sigmas)
        model_config = hparams.model
        self.layers = model_config.layers

    def training_step(self, batch, _) -> dict:  # pylint: disable=arguments-differ
        self.model.teacher_model.eval()
        teacher_features, student_features = self.model.forward(batch["image"])
        env_labels = self.get_env_labels(batch)
        loss = self.loss(teacher_features, student_features)
        for layer in student_features.keys():
            s_feature = student_features[layer]
            loss += self.alpha * self.mmd_loss(env_labels, s_feature, sum_of_distances=False,
                                               sigmas=self.sigmas,
                                               normalize_per_channel=False)
        self.log("train_loss", loss)
        return {"loss": loss}
