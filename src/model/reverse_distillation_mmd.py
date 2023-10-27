import torch
from anomalib.models.reverse_distillation import ReverseDistillationLightning
from src.loss_functions import MMDLoss
from .dg_module import DgLightning


class ReverseDistillationMmdLightning(ReverseDistillationLightning, DgLightning):
    def __init__(self, hparams):
        ReverseDistillationLightning.__init__(self, hparams)
        DgLightning.__init__(self, pooled=True, dataset_format=hparams.dataset.format,
                             dataset_name=hparams.dataset.name, alpha=hparams.model.alpha,
                             sigmas=hparams.model.sigmas)
        # self.mmd_loss = MMDLoss(pooled=False)
    def training_step(self, batch, *args, **kwargs):
        """Training Step of Reverse Distillation Model.

        Features are extracted from three layers of the Encoder model. These are passed to the bottleneck layer
        that are passed to the decoder network. The loss is then calculated based on the cosine similarity between the
        encoder and decoder features.

        Args:
          batch (batch: dict[str, str | Tensor]): Input batch

        Returns:
          Feature Map
        """
        del args, kwargs  # These variables are not used.
        encoder_features, decoder_features = self.model(batch["image"])
        env_labels = self.get_env_labels(batch)
        loss = self.loss(encoder_features=encoder_features, decoder_features=decoder_features)
        for df in decoder_features:
            """
            [batch_size, channel, width, height] = df.shape
            # assume all divisible by 16
            step_w = 16
            step_h = 16
            loss_mmd = 0.0
            counter = 0
            for w in range(0, width, step_w):
                for h in range(0, height, step_h):
                    loss_mmd += self.mmd_loss(env_labels, df[:, :, w: w+step_w, h: h+step_h],
                                                        normalize_per_channel=False, sum_of_distances=False, sigmas=self.sigmas)
                    counter += 1
            loss_mmd /= counter
            loss += (self.alpha*loss_mmd)/(len(decoder_features))
            """
            loss += self.alpha*self.mmd_loss(env_labels, df, normalize_per_channel=False, 
                    sum_of_distances=False, sigmas=self.sigmas)
        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}