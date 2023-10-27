import torch
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch import Tensor, optim
from src.model import Model

class MeanShiftedLightning(pl.LightningModule):
    def __init__(self, backbone):
        super(MeanShiftedCL, self).__init__()
        self.model = utils.Model(backbone='resnet18',pretrained=True,)
        self.center = None
        self.learning_rate = backbone['lr']
        
    def forward(self, x):
        return self.model(x)

    def contrastive_loss(self, out_1, out_2):
        out_1 = F.normalize(out_1, dim=-1)
        out_2 = F.normalize(out_2, dim=-1)
        bs = out_1.size(0)
        temp = 0.25
        out = torch.cat([out_1, out_2], dim=0)
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temp)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * bs, device=sim_matrix.device)).bool()
        sim_matrix = sim_matrix.masked_select(mask).view(2 * bs, -1)
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temp)
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        return loss
    
    def on_train_start(self):
        train_feature_space = []
        with torch.no_grad():
            for batch in self.train_dataloader():
                imgs = batch["image"]
                imgs = imgs.to(self.device)
                features = self.model(imgs)
                train_feature_space.append(features)
            train_feature_space = torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()
        self.center = torch.FloatTensor(train_feature_space).mean(dim=0)

    def configure_optimizers(self):
        decoders_parameters = []
        for decoder_idx in range(len(self.model.pool_layers)):
            decoders_parameters.extend(list(self.model.decoders[decoder_idx].parameters()))

        optimizer = optim.Adam(
            params=decoders_parameters,
            lr=self.learning_rate,
        )
        return optimizer
    
    def _run_epoch(self, batch):
        (img1, img2), y = batch[:2]
        out_1 = self.model(img1)
        out_2 = self.model(img2)
        out_1 = out_1 - self.center
        out_2 = out_2 - self.center
        loss = 0
        loss += contrastive_loss(out_1, out_2)# * 1e-10
        self.total_num += img1.size(0)
        self.total_loss += loss.item() * img1.size(0)
        return loss

    def training_step(self, batch, batch_idx):
        images = batch["image"]
        avg_loss = self._run_epoch(images)       
        
        self.log('train_loss', avg_loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": avg_loss}

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        features = self.model(imgs)
        return features, labels

    def validation_epoch_end(self, outputs):
        features = torch.cat([output[0] for output in outputs], dim=0).contiguous().cpu().numpy()
        labels = torch.cat([output[1] for output in outputs], dim=0).cpu().numpy()
        distances = utils.knn_score(self.center, features)
        auc = roc_auc_score(labels, distances)
        self.log('val_auc', auc)
        return {'val_auc': auc}

    def on_fit_start(self):
        feature_space = []
        for (imgs, _) in self.train_dataloader():
            features = self.model(imgs)
            feature_space.append(features)
        feature_space = torch.cat(feature_space, dim=0).contiguous().cpu
