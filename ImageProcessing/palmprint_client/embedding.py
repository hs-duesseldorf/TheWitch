from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

CHECKPOINT_FORMAT = "palmprint_resnet18_embedding_v1"


class ResNet18EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim: int = 256, *, pretrained: bool = True):
        super().__init__()
        weights = None
        if pretrained:
            try:
                weights = models.ResNet18_Weights.DEFAULT
            except Exception:
                weights = None

        backbone = models.resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.embedding = nn.Sequential(
            nn.Linear(in_features, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        embedding = self.embedding(features)
        return F.normalize(embedding, dim=1)


class ArcMarginProduct(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        scale: float = 32.0,
        margin: float = 0.35,
        easy_margin: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=0.0, max=1.0))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return logits * self.scale


def build_embedding_checkpoint(
    model: ResNet18EmbeddingNet,
    *,
    num_classes: int,
    epoch: int,
    image_size: int,
    train_sessions: tuple[str, ...],
    val_sessions: tuple[str, ...],
    metrics: dict[str, Any],
    classifier_state: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    return {
        "format": CHECKPOINT_FORMAT,
        "model_name": "resnet18",
        "embedding_dim": int(model.embedding[0].out_features),
        "num_classes": int(num_classes),
        "image_size": int(image_size),
        "input_mode": "palm_gray",
        "metric": "cosine",
        "epoch": int(epoch),
        "train_sessions": list(train_sessions),
        "val_sessions": list(val_sessions),
        "metrics": metrics,
        "model_state": model.state_dict(),
        "classifier_state": classifier_state,
    }


def load_embedding_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[ResNet18EmbeddingNet, dict[str, Any]]:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{checkpoint_path} is not a {CHECKPOINT_FORMAT} checkpoint")

    embedding_dim = int(checkpoint.get("embedding_dim", 256))
    model = ResNet18EmbeddingNet(embedding_dim=embedding_dim, pretrained=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model = model.eval().to(device)
    return model, checkpoint
