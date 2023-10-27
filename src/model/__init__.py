from .stfpm_mmd import StfpmMmdLightning
from .cflow_mmd import CflowMmdLightning
from .fastflow_mmd import FastflowMmdLightning
from .reverse_distillation_mmd import ReverseDistillationMmdLightning
from .cfa_mmd import CfaMmdLightning
from .mean_shifted_mmd import MeanShiftedLightning
from torchvision import models

class Model(torch.nn.Module):
    def __init__(self,
                 invariant,
                 backbone,
                 checkpoint_path=None):
        super().__init__()
        if invariant is True:
            # Code for loading the checkpoint
            self.load_state_dict(torch.load(checkpoint_path))  # add this line
        else: 
            if backbone == 152:
                self.backbone = models.resnet152(pretrained=True)
            elif backbone == 50:
                self.backbone = models.resnet50(pretrained=True)
            else:
                self.backbone = models.resnet18(pretrained=True)
            self.backbone.fc = torch.nn.Identity()
            freeze_parameters(self.backbone, backbone, train_fc=False)

    def forward(self, x):
        z1 = self.backbone(x)
        z_n = F.normalize(z1, dim=-1)
        return z_n

def freeze_parameters(model, backbone, train_fc=False):
    if not train_fc:
        for p in model.fc.parameters():
            p.requires_grad = False
    if backbone == 152:
        for p in model.conv1.parameters():
            p.requires_grad = False
        for p in model.bn1.parameters():
            p.requires_grad = False
        for p in model.layer1.parameters():
            p.requires_grad = False
        for p in model.layer2.parameters():
            p.requires_grad = False
