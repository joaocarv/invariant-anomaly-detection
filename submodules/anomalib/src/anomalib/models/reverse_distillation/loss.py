"""Loss function for Reverse Distillation."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from torch import Tensor, nn


class ReverseDistillationLoss(nn.Module):
    """Loss function for Reverse Distillation."""

    def loss(self, ef, df):
        mse_loss = torch.nn.MSELoss(reduction="sum")
        ef = ef.reshape(ef.shape[0], -1)
        df = df.reshape(df.shape[0], -1)
        return mse_loss(torch.nn.functional.normalize(ef, dim=1), torch.nn.functional.normalize(df, dim=1))

    def forward(self, encoder_features: list[Tensor], decoder_features: list[Tensor]) -> Tensor:
        """Computes cosine similarity loss based on features from encoder and decoder.

        Args:
            encoder_features (list[Tensor]): List of features extracted from encoder
            decoder_features (list[Tensor]): List of features extracted from decoder

        Returns:
            Tensor: Loss
        """

        losses = list(map(self.loss, encoder_features, decoder_features))
        loss_sum = 0
        for loss in losses:
            loss_sum += loss
        return loss_sum
