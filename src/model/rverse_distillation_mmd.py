from anomalib.models.reverse_distillation import ReverseDistillationLightning
from .dg_module import DgLightning


class ReverseDistillationMmdLightning(ReverseDistillationLightning, DgLightning):
    def __init__(self, hparams):
        ReverseDistillationLightning.__init__(self, hparams)
        DgLightning.__init__(self, pooled=True, dataset_format=hparams.dataset.format,
                             dataset_name=hparams.dataset.name, alpha=hparams.model.alpha,
                             sigmas=hparams.model.sigmas)

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
            loss += self.alpha*self.mmd_loss(env_labels, df, normalize_per_channel=False, sum_of_distances=False,
                                             sigmas=self.sigmas)
        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}
