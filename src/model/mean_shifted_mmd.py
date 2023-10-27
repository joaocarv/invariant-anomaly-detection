import torch
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
import utils
from torch.optim import SGD
from torch.utils.data import DataLoader


import torch
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.model import Model
from src.model.mean_shifted import MeanShiftedLightning

class MeanShiftedMMDLightning(MeanShiftedLightning, DgLightning):

    def __init__(
            self,
            hparams: DictConfig
    ):
        MeanShiftedLightning.__init__(self, hparams)
        DgLightning.__init__(self, pooled=True, dataset_format=hparams.dataset.format,
                             dataset_name=hparams.dataset.name, alpha=hparams.model.alpha,
                             sigmas=hparams.model.sigmas)
        
        model_config = hparams.model

    def training_step(self, batch, _) -> dict:  # pylint: disable=arguments-differ
        
        images = batch["image"]
        avg_loss = self._run_epoch(images)
        self.log('train_loss', avg_loss.item(),on_epoch=True, prog_bar=True, logger=True)
        
        loss += self.alpha*self.mmd_loss(env_labels, target_features, sum_of_distances=False,
                                    sigmas=self.sigmas, normalize_per_channel=False)

        self.log("train_loss", loss)
        return {"loss": loss}
