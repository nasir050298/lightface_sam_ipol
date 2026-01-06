"""LightFace-SAM v2 model definition (embedder + optional margin head).

IPOL demo uses inference only:
  aligned 112x112 RGB -> 256-d L2-normalized embedding

The checkpoint contains classifier weights (margin cosine head) from training.
For inference, we only need backbone + low-rank embedding head, but we load the
full checkpoint and ignore the classifier during forward() when labels=None.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class LowRankHead(nn.Module):
    """Low-rank embedding head: 1280 -> 128 -> 256."""
    def __init__(self, in_features: int = 1280, bottleneck: int = 128, out_features: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(in_features, bottleneck, bias=False)
        self.fc2 = nn.Linear(bottleneck, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        return x


class MarginCosineProduct(nn.Module):
    """CosFace-style margin-based cosine classifier (training-time only)."""
    def __init__(self, in_features: int, num_classes: int, s: float = 30.0, m: float = 0.2):
        super().__init__()
        self.s = float(s)
        self.m = float(m)
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        # x is expected to be L2-normalized
        W = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(x, W)

        if labels is None:
            return self.s * cosine

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = (cosine - one_hot * self.m) * self.s
        return logits


class LightFaceSAMV2(nn.Module):
    """MobileNetV2 backbone + low-rank embedding head (+ optional margin head)."""
    def __init__(self, num_classes: int, bottleneck: int = 128, emb_dim: int = 256,
                 s: float = 30.0, m: float = 0.2, imagenet_pretrained: bool = False):
        super().__init__()

        # For IPOL demo we don't need ImageNet weights; checkpoint overwrites parameters.
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if imagenet_pretrained else None
        self.backbone = mobilenet_v2(weights=weights)
        self.backbone.classifier[1] = nn.Identity()

        self.embedding_head = LowRankHead(in_features=1280, bottleneck=bottleneck, out_features=emb_dim)
        self.margin_fc = MarginCosineProduct(in_features=emb_dim, num_classes=num_classes, s=s, m=m)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None):
        feats = self.backbone(x)              # [B, 1280]
        emb = self.embedding_head(feats)      # [B, 256]
        emb = F.normalize(emb, p=2, dim=1)    # L2-normalized embeddings

        if labels is None:
            return emb                        # inference path

        logits = self.margin_fc(emb, labels)  # training path
        return logits, emb


def infer_num_classes_from_ckpt(state_dict: dict) -> int:
    """Infer num_classes from classifier weight matrix in the checkpoint."""
    w = state_dict.get("margin_fc.weight", None)
    if w is None:
        raise KeyError("Checkpoint missing 'margin_fc.weight' needed to infer num_classes.")
    return int(w.shape[0])
