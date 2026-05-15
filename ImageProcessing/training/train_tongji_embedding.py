#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms

from ImageProcessing.palmprint_client.constants import IMAGENET_MEAN, IMAGENET_STD, MODEL_RESIZE_SIZE
from ImageProcessing.palmprint_client.embedding import (
    ArcMarginProduct,
    ResNet18EmbeddingNet,
    build_embedding_checkpoint,
)
from ImageProcessing.palmprint_client.utils import set_deterministic

IMAGES_PER_PALM_PER_SESSION = 10
PALMS_PER_PERSON = 2
DEFAULT_DATASET_ROOT = Path("assets/datasets/tongji/roi")
TRAINING_LOSSES = ("arcface", "contrastive")
LABEL_MODES = ("person", "palm")
EPOCHS = 8
BATCH_SIZE = 128
EMBEDDING_DIM = 256
IMAGE_SIZE = 224
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
CONTRASTIVE_MARGIN = 0.0
SAMPLES_PER_CLASS = 2
NUM_WORKERS = 4
SEED = 1234


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Tongji palmprint embedding model")
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--loss", choices=TRAINING_LOSSES, default="arcface")
    parser.add_argument("--label_mode", choices=LABEL_MODES, default="person")
    args = parser.parse_args()
    if args.output is None:
        args.output = default_output_path(args)
    return args


def default_output_path(args: argparse.Namespace) -> Path:
    return Path(
        f"assets/weights/tongji_resnet18_{args.loss}_{args.label_mode}_{EMBEDDING_DIM}d.pt"
    )


def infer_label_from_path(path: Path, label_mode: str) -> int:
    image_index = int(path.stem)
    if label_mode == "person":
        images_per_class = IMAGES_PER_PALM_PER_SESSION * PALMS_PER_PERSON
    elif label_mode == "palm":
        images_per_class = IMAGES_PER_PALM_PER_SESSION
    else:
        raise ValueError(f"Unsupported label mode: {label_mode}")

    label = (image_index - 1) // images_per_class
    if label < 0:
        raise ValueError(f"Invalid Tongji filename: {path.name}")
    return label


def pil_to_enhanced_gray(image: Image.Image) -> Image.Image:
    import cv2

    from ImageProcessing.palmprint_client.preprocessing import enhance_palm_roi

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


def repeat_gray_to_rgb(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.repeat(3, 1, 1)


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
                transforms.Lambda(repeat_gray_to_rgb),
                transforms.Normalize(
                    mean=IMAGENET_MEAN,
                    std=IMAGENET_STD,
                ),
            ]
        )

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image.convert("L"))


class TongjiPalmDataset(Dataset):
    def __init__(self, root: Path, sessions: tuple[str, ...], transform, *, label_mode: str):
        self.root = root
        self.sessions = tuple(sessions)
        self.transform = transform
        self.label_mode = label_mode
        self.samples: list[tuple[Path, int]] = []

        for session in self.sessions:
            session_dir = root / session
            if not session_dir.exists():
                raise FileNotFoundError(f"Missing session directory: {session_dir}")
            for path in sorted(session_dir.glob("*.bmp")):
                self.samples.append((path, infer_label_from_path(path, label_mode)))

        if not self.samples:
            raise RuntimeError(f"No BMP files found under {root}")

        self.labels = [label for _, label in self.samples]
        self.num_classes = max(label for _, label in self.samples) + 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            return self.transform(image), label


class PositivePairBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        labels: list[int],
        *,
        batch_size: int,
        samples_per_class: int,
        seed: int,
    ):
        if samples_per_class < 2:
            raise ValueError("samples_per_class must be >= 2 for contrastive training")
        if batch_size < samples_per_class:
            raise ValueError("batch_size must be >= samples_per_class")

        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        self.seed = seed
        self.iteration = 0
        self.classes_per_batch = max(1, batch_size // samples_per_class)
        self.label_to_indices: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            self.label_to_indices[int(label)].append(index)
        self.labels = sorted(self.label_to_indices.keys())
        self.batch_count = max(1, len(labels) // (self.classes_per_batch * samples_per_class))

    def __iter__(self):
        rng = random.Random(self.seed + self.iteration)
        self.iteration += 1
        for _ in range(self.batch_count):
            if len(self.labels) <= self.classes_per_batch:
                selected_labels = list(self.labels)
                rng.shuffle(selected_labels)
            else:
                selected_labels = rng.sample(self.labels, self.classes_per_batch)

            batch: list[int] = []
            for label in selected_labels:
                candidates = self.label_to_indices[label]
                if len(candidates) >= self.samples_per_class:
                    batch.extend(rng.sample(candidates, self.samples_per_class))
                else:
                    batch.extend(rng.choices(candidates, k=self.samples_per_class))
            rng.shuffle(batch)
            yield batch[: self.batch_size]

    def __len__(self) -> int:
        return self.batch_count


def build_loader(
    dataset: TongjiPalmDataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    positive_pairs: bool = False,
    samples_per_class: int = 2,
    seed: int = 0,
) -> DataLoader:
    if positive_pairs:
        return DataLoader(
            dataset,
            batch_sampler=PositivePairBatchSampler(
                dataset.labels,
                batch_size=batch_size,
                samples_per_class=samples_per_class,
                seed=seed,
            ),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=shuffle,
    )


class ContrastiveCosineLoss(nn.Module):
    def __init__(self, *, margin: float = 0.0):
        super().__init__()
        self.loss = nn.CosineEmbeddingLoss(margin=margin)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = nn.functional.normalize(embeddings, dim=1)
        labels = labels.view(-1)
        if embeddings.size(0) < 2:
            return embeddings.sum() * 0.0

        left, right = torch.triu_indices(embeddings.size(0), embeddings.size(0), offset=1, device=embeddings.device)
        targets = torch.where(labels[left] == labels[right], 1.0, -1.0).to(embeddings.device)
        return self.loss(embeddings[left], embeddings[right], targets)


def embed_dataset(
    model: nn.Module,
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
    model: nn.Module,
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
    set_deterministic(SEED)
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    train_dataset = TongjiPalmDataset(
        args.dataset_root,
        sessions=("session1",),
        transform=PalmTrainTransform(IMAGE_SIZE),
        label_mode=args.label_mode,
    )
    gallery_dataset = TongjiPalmDataset(
        args.dataset_root,
        sessions=("session1",),
        transform=PalmEvalTransform(IMAGE_SIZE),
        label_mode=args.label_mode,
    )
    probe_dataset = TongjiPalmDataset(
        args.dataset_root,
        sessions=("session2",),
        transform=PalmEvalTransform(IMAGE_SIZE),
        label_mode=args.label_mode,
    )

    train_loader = build_loader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
        positive_pairs=args.loss == "contrastive",
        samples_per_class=SAMPLES_PER_CLASS,
        seed=SEED,
    )
    gallery_loader = build_loader(
        gallery_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
    )
    probe_loader = build_loader(
        probe_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
    )

    model = ResNet18EmbeddingNet(embedding_dim=EMBEDDING_DIM, pretrained=True).to(device)
    margin_head = ArcMarginProduct(EMBEDDING_DIM, train_dataset.num_classes).to(device) if args.loss == "arcface" else None
    criterion = nn.CrossEntropyLoss() if args.loss == "arcface" else ContrastiveCosineLoss(margin=CONTRASTIVE_MARGIN)

    optimizer_params = [{"params": model.parameters(), "lr": LEARNING_RATE}]
    if margin_head is not None:
        optimizer_params.append({"params": margin_head.parameters(), "lr": LEARNING_RATE})

    optimizer = AdamW(optimizer_params, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler(device.type, enabled=use_amp)

    history: list[dict[str, float]] = []
    best_probe_top1 = -1.0
    best_epoch = -1
    started_at = time.time()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Training on {device.type} with {len(train_dataset)} train images, "
        f"{len(probe_dataset)} probe images, {train_dataset.num_classes} classes, "
        f"loss={args.loss}, label_mode={args.label_mode}."
    )

    for epoch in range(EPOCHS):
        model.train()
        if margin_head is not None:
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
                if margin_head is None:
                    loss = criterion(embeddings, labels)
                else:
                    loss = criterion(margin_head(embeddings, labels), labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = images.size(0)
            samples_seen += batch_size
            running_loss += float(loss.item()) * batch_size

            if step % 20 == 0 or step == len(train_loader):
                avg_loss = running_loss / max(samples_seen, 1)
                print(
                    f"epoch {epoch + 1:02d}/{EPOCHS:02d} "
                    f"step {step:03d}/{len(train_loader):03d} "
                    f"loss {avg_loss:.4f}"
                )

        scheduler.step()

        train_loss = running_loss / max(samples_seen, 1)
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
                "seed": int(SEED),
                "device": str(device),
                "trained_at_unix": int(time.time()),
                "model": "resnet18",
                "loss": args.loss,
                "label_mode": args.label_mode,
            }
            checkpoint = build_embedding_checkpoint(
                model,
                num_classes=train_dataset.num_classes,
                epoch=best_epoch,
                image_size=IMAGE_SIZE,
                train_sessions=("session1",),
                val_sessions=("session2",),
                metrics=metrics,
                classifier_state=margin_head.state_dict() if margin_head is not None else None,
                model_name="resnet18",
                training_objective=args.loss,
                label_mode=args.label_mode,
                training_config={
                    "contrastive_margin": float(CONTRASTIVE_MARGIN),
                    "samples_per_class": int(SAMPLES_PER_CLASS),
                },
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
