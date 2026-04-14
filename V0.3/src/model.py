import torch
import torch.nn as nn
import timm

from src.config import cfg, CLASS_SCHEME

class MultimodalSkinCancerModel(nn.Module):
    """
    Multimodal skin cancer classification model combining image and clinical metadata.

    Architecture:
        - Image backbone: EfficientNet-Bx (pretrained, optionally frozen) -> X-d features
        - (B0: 1280, B1: 1280, B2: 1408, B3: 1536, B4: 1792, B5: 2048, B6: 2304, B7: 2560) | Dimensional pooled features
        - (B0: 224, B1: 240, B2: 260, B3: 300, B4: 380, B5: 456, B6: 528, B7: 600) | Native img size.

        - Metadata branch: MLP (metadata_dim -> 64 -> 32) with ReLU + Dropout
        - Fusion: Concatenate image + metadata features (X + 32 = X+32-d)
        - Classifier: MLP (X -> 512 -> num_classes) with ReLU + Dropout

    Parameters:
        num_classes (int): Number of output classes. Default: 8.
        metadata_dim (int): Dimensionality of the metadata feature vector. Default: 23 (1 for Age, 1 for Sex, 21 for Localization).
        freeze_backbone (bool): Whether to freeze backbone weights initially. Default: True.
    """

    def __init__(self, num_classes=len(CLASS_SCHEME), metadata_dim=23, freeze_backbone=cfg.model.freeze_backbone):
        super().__init__()

        # Image backbone (EfficientNet-B0 pretrained on ImageNet, classifier head removed)
        self.backbone = timm.create_model(cfg.model.backbone, pretrained=cfg.model.pretrained, num_classes=0)
        backbone_dim = self.backbone.num_features  

        # Freeze backbone for initial warmup training (only classifier + meta branch train)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Metadata branch: small MLP to process clinical features (age, sex, localization)
        self.meta_branch = nn.Sequential(
            nn.Linear(metadata_dim, cfg.model.meta_hidden[0]),
            nn.ReLU(),
            nn.Dropout(cfg.model.meta_dropout),
            nn.Linear(cfg.model.meta_hidden[0], cfg.model.meta_hidden[1]),
            nn.ReLU()
        )

        # Fusion classifier: combines image and metadata representations
        self.classifier = nn.Sequential(
            nn.Linear(backbone_dim + cfg.model.meta_hidden[1], cfg.model.classifier_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.model.classifier_dropout),
            nn.Linear(cfg.model.classifier_hidden, num_classes)
        )

    def forward(self, image, metadata):
        img_features = self.backbone(image)           
        meta_features = self.meta_branch(metadata)    
        fused = torch.cat([img_features, meta_features], dim=1)  
        return self.classifier(fused)                 

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters to enable end-to-end fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
