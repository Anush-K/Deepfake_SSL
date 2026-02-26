import torch.nn as nn
import timm
import torch.nn.functional as F
from ssl_simclr.gem import GeM
from ssl_simclr.projection_head import ProjectionHead


class SSLModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=True,
            num_classes=0
        )

        self.pool = GeM()
        self.projector = ProjectionHead(1792, 512, 128)

    def forward(self, x):

        features = self.backbone.forward_features(x)
        pooled = self.pool(features).flatten(1)

        proj = self.projector(pooled)
        proj = F.normalize(proj, dim=1)

        return proj