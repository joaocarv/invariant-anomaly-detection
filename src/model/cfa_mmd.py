import torch

from anomalib.models.cfa import CfaLightning
from .dg_module import DgLightning


class CfaMmdLightning(CfaLightning, DgLightning):
    def __init__(self, hparams):
        CfaLightning.__init__(self, hparams)
        DgLightning.__init__(self, pooled=True, dataset_format=hparams.dataset.format,
                             dataset_name=hparams.dataset.name, alpha=hparams.model.alpha,
                             sigmas=hparams.model.sigmas)

    def training_step(self, batch: dict, *args, **kwargs):
        """Training step for the CFA model.

        Args:
            batch (dict[str, str | Tensor]): Batch input.

        Returns:
            STEP_OUTPUT: Loss value.
        """
        del args, kwargs  # These variables are not used.

        distance, target_features = self.model(batch["image"])
        env_labels = self.get_env_labels(batch)
        loss = self.loss(distance)
        loss += self.alpha*self.mmd_loss(env_labels, target_features, sum_of_distances=False,
                                         sigmas=self.sigmas, normalize_per_channel=False)
        return {"loss": loss}
