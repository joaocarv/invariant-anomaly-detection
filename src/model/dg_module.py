import pytorch_lightning as pl
from src.loss_functions import MMDLoss


class DgLightning:
    def __init__(self, pooled, dataset_format, dataset_name, alpha, sigmas):
        self.mmd_loss = MMDLoss(pooled=pooled)
        self.dataset_format = dataset_format
        self.dataset_name = dataset_name
        self.alpha = alpha
        self.sigmas = sigmas

    def get_env_labels(self, batch):
        return batch["env_label"]
