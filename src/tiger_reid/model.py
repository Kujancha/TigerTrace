import timm
import torch.nn as nn
import torch.nn.functional as F

BACKBONE = "vit_base_patch16_224.augreg_in21k_ft_in1k"


class TigerViT(nn.Module):
    def __init__(self, pretrained=True, blocks=4, checkpointing=False):
        super().__init__()
        if not 0 <= blocks <= 12:
            raise ValueError("Trainable blocks must be 0..12.")
        self.backbone = timm.create_model(
            BACKBONE, pretrained=pretrained, num_classes=0
        )
        self.embedding_head = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 128)
        )
        for param in self.backbone.parameters():
            param.requires_grad = False
        if blocks:
            for block in self.backbone.blocks[-blocks:]:
                for param in block.parameters():
                    param.requires_grad = True
            for param in self.backbone.norm.parameters():
                param.requires_grad = True
        if checkpointing:
            self.backbone.set_grad_checkpointing(True)

    def features(self, x):
        return F.normalize(self.backbone(x), dim=1)

    def forward(self, x):
        return F.normalize(self.embedding_head(self.backbone(x)), dim=1)
