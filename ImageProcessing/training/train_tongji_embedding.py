#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from palmprint_client.constants import IMAGENET_MEAN, IMAGENET_STD, MODEL_RESIZE_SIZE
from palmprint_client.embedding import ArcMarginProduct, ResNet18EmbeddingNet, build_embedding_checkpoint
from palmprint_client.preprocessing import enhance_palm_roi
from palmprint_client.utils import set_deterministic

IMAGES_PER_CLASS_PER_SESSION = 10
DEFAULT_DATASET_ROOT = Path("assets/datasets/tongji/roi")
DEFAULT_OUTPUT_PATH = Path("assets/weights/tongji_resnet18_arcface_256d.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Tongji palmprint embedding model with ArcFace")
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def infer_label_from_path(path: Path) -> int:
    image_index = int(path.stem)
    label = (image_index - 1) // IMAGES_PER_CLASS_PER_SESSION
    if label < 0:
        raise ValueError(f"Invalid Tongji filename: {path.name}")
    return label


def pil_to_enhanced_gray(image: Image.Image) -> Image.Image:
    gray = np.array(image.convert("L"), dtype=np.uint8)
    enhanced = enhance_palm_roi(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    return Image.fromarray(enhanced, mode="L")


class RandomGamma:
    def __init__(self, low: float = 0.75, high: float = 1.35):
        self.low = low
        self.high = high

    def __call__(self, image: Image.Image) -> Image.Image:
        gamma = random.uniform(self.low, self.high)
        array = np.array(image, dtype=np.float32) / 255.0
        corrected = np.power(np.clip(array, 0.0, 1.0), gamma)
        return Image.fromarray(np.clip(corrected * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")


class PalmTrainTransform:
    def __init__(self, image_size: int):
        self.image_size = image_size
        self.random_resized_crop = transforms.RandomResizedCrop(
            image_size,
            scale=(0.88, 1.0),
            ratio=(0.96, 1.04),
            interpolation=transforms.InterpolationMode.BILINEAR,
        )
        self.random_affine = transforms.RandomAffine(
            degrees=7,
            translate=(0.03, 0.03),
            scale=(0.97, 1.03),
            shear=3,
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=0,
        )
        self.random_perspective = transforms.RandomPerspective(
            distortion_scale=0.12,
            p=0.15,
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=0,
        )
        self.blur = transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.2))
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )
        self.gamma = RandomGamma()

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("L")
        if random.random() < 0.7:
            image = pil_to_enhanced_gray(image)
        if random.random() < 0.6:
            image = self.gamma(image)
        image = ImageOps.autocontrast(image)
        image = image.resize((MODEL_RESIZE_SIZE, MODEL_RESIZE_SIZE), Image.Resampling.BILINEAR)
        image = self.random_affine(image)
        image = self.random_perspective(image)
        image = self.random_resized_crop(image)
        if random.random() < 0.2:
            image = self.blur(image)

        tensor = self.to_tensor(image).repeat(3, 1, 1)
        return self.normalize(tensor)


class PalmEvalTransform:
    def __init__(self, image_size: int):
        self.transform = transforms.Compose(
            [
                transforms.Resize(MODEL_RESIZE_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Lambda(lambda tensor: tensor.repeat(3, 1, 1)),
                transforms.Normalize(
                    mean=IMAGENET_MEAN,
                    std=IMAGENET_STD,
                ),
            ]
        )

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image.convert("L"))


class TongjiPalmDataset(Dataset):
    def __init__(self, root: Path, sessions: tuple[str, ...], transform):
        self.root = root
        self.sessions = tuple(sessions)
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        for session in self.sessions:
            session_dir = root / session
            if not session_dir.exists():
                raise FileNotFoundError(f"Missing session directory: {session_dir}")
            for path in sorted(session_dir.glob("*.bmp")):
                self.samples.append((path, infer_label_from_path(path)))

        if not self.samples:
            raise RuntimeError(f"No BMP files found under {root}")

        self.num_classes = max(label for _, label in self.samples) + 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            return self.transform(image), label


def build_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=shuffle,
    )


def embed_dataset(
    model: ResNet18EmbeddingNet,
    loader: DataLoader,
    *,
    device: torch.device,
    use_amp: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []

    model.eval()
    with torch.inference_mode():
        for images, batch_labels in loader:
            images = images.to(device, non_blocking=True)
            with autocast(device_type=device.type, enabled=use_amp):
                batch_embeddings = model(images)
            embeddings.append(batch_embeddings.float().cpu())
            labels.append(batch_labels.clone())

    return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0)


def gallery_centroids(embeddings: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    buckets: dict[int, list[torch.Tensor]] = defaultdict(list)
    for embedding, label in zip(embeddings, labels):
        buckets[int(label)].append(embedding)

    centroids = []
    for label in range(num_classes):
        class_embeddings = torch.stack(buckets[label], dim=0)
        centroid = nn.functional.normalize(class_embeddings.mean(dim=0), dim=0)
        centroids.append(centroid)
    return torch.stack(centroids, dim=0)


def evaluate_probe_top1(
    model: ResNet18EmbeddingNet,
    gallery_loader: DataLoader,
    probe_loader: DataLoader,
    *,
    num_classes: int,
    device: torch.device,
    use_amp: bool,
) -> float:
    gallery_embeddings, gallery_labels = embed_dataset(model, gallery_loader, device=device, use_amp=use_amp)
    probe_embeddings, probe_labels = embed_dataset(model, probe_loader, device=device, use_amp=use_amp)

    centroids = gallery_centroids(gallery_embeddings, gallery_labels, num_classes=num_classes)
    scores = probe_embeddings @ centroids.T
    predictions = scores.argmax(dim=1)
    accuracy = (predictions == probe_labels).float().mean().item()
    return accuracy


def format_duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    remainder = int(seconds % 60)
    return f"{minutes:02d}m{remainder:02d}s"


def main() -> None:
    args = parse_args()
    set_deterministic(args.seed)
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    train_dataset = TongjiPalmDataset(
        args.dataset_root,
        sessions=("session1",),
        transform=PalmTrainTransform(args.image_size),
    )
    gallery_dataset = TongjiPalmDataset(
        args.dataset_root,
        sessions=("session1",),
        transform=PalmEvalTransform(args.image_size),
    )
    probe_dataset = TongjiPalmDataset(
        args.dataset_root,
        sessions=("session2",),
        transform=PalmEvalTransform(args.image_size),
    )

    train_loader = build_loader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    gallery_loader = build_loader(
        gallery_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    probe_loader = build_loader(
        probe_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    model = ResNet18EmbeddingNet(embedding_dim=args.embedding_dim, pretrained=True).to(device)
    margin_head = ArcMarginProduct(args.embedding_dim, train_dataset.num_classes).to(device)

    optimizer = AdamW(
        [
            {"params": model.parameters(), "lr": args.lr},
            {"params": margin_head.parameters(), "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(device.type, enabled=use_amp)

    history: list[dict[str, float]] = []
    best_probe_top1 = -1.0
    best_epoch = -1
    started_at = time.time()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Training on {device.type} with {len(train_dataset)} train images, "
        f"{len(probe_dataset)} probe images, {train_dataset.num_classes} classes."
    )

    for epoch in range(args.epochs):
        model.train()
        margin_head.train()

        running_loss = 0.0
        samples_seen = 0

        epoch_started = time.time()
        for step, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=use_amp):
                embeddings = model(images)
                logits = margin_head(embeddings, labels)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = images.size(0)
            samples_seen += batch_size
            running_loss += float(loss.item()) * batch_size

            if step % 20 == 0 or step == len(train_loader):
                avg_loss = running_loss / max(samples_seen, 1)
                print(
                    f"epoch {epoch + 1:02d}/{args.epochs:02d} "
                    f"step {step:03d}/{len(train_loader):03d} "
                    f"loss {avg_loss:.4f}"
                )

        scheduler.step()

        train_loss = running_loss / len(train_dataset)
        probe_top1 = evaluate_probe_top1(
            model,
            gallery_loader,
            probe_loader,
            num_classes=train_dataset.num_classes,
            device=device,
            use_amp=use_amp,
        )

        epoch_metrics = {
            "epoch": float(epoch + 1),
            "train_loss": float(train_loss),
            "probe_top1": float(probe_top1),
        }
        history.append(epoch_metrics)

        epoch_seconds = time.time() - epoch_started
        print(
            f"epoch {epoch + 1:02d} complete "
            f"loss={train_loss:.4f} probe_top1={probe_top1:.4f} "
            f"time={format_duration(epoch_seconds)}"
        )

        if probe_top1 > best_probe_top1:
            best_probe_top1 = probe_top1
            best_epoch = epoch + 1
            metrics = {
                "best_probe_top1": float(best_probe_top1),
                "best_epoch": int(best_epoch),
                "history": history,
                "seed": int(args.seed),
                "device": str(device),
                "trained_at_unix": int(time.time()),
            }
            checkpoint = build_embedding_checkpoint(
                model,
                num_classes=train_dataset.num_classes,
                epoch=best_epoch,
                image_size=args.image_size,
                train_sessions=("session1",),
                val_sessions=("session2",),
                metrics=metrics,
                classifier_state=margin_head.state_dict(),
            )
            torch.save(checkpoint, str(args.output))
            args.output.with_suffix(".json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            print(f"saved checkpoint to {args.output} at probe_top1={best_probe_top1:.4f}")

    total_seconds = time.time() - started_at
    print(
        f"training finished in {format_duration(total_seconds)} "
        f"best_epoch={best_epoch} best_probe_top1={best_probe_top1:.4f}"
    )


if __name__ == "__main__":
    main()
