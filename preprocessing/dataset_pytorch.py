"""
PyTorch Dataset and DataLoader module for Radar Spectrogram Image Classification.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import config

logger = logging.getLogger(__name__)

# Standard ImageNet normalization for PyTorch vision models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_default_transforms(
    is_train: bool = True,
    target_shape: Tuple[int, int] = config.SPECTROGRAM_SHAPE,
) -> transforms.Compose:
    """
    Construct torchvision preprocessing and augmentation pipeline.

    Parameters
    ----------
    is_train : bool
        Whether to apply data augmentations (flips, color jitter).
    target_shape : Tuple[int, int]
        (height, width) target image size.

    Returns
    -------
    transforms.Compose
        Torchvision transform pipeline.
    """
    transform_list = [
        transforms.Resize(target_shape),
    ]

    if is_train:
        transform_list.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
            ]
        )

    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    return transforms.Compose(transform_list)


class RadarSpectrogramDataset(Dataset):
    """
    PyTorch Dataset for Radar Spectrogram Images.
    """

    def __init__(
        self,
        samples: List[Dict[str, Union[str, int]]],
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Parameters
        ----------
        samples : List[Dict]
            List of dictionary items containing 'image_path' and 'class_id'.
        transform : Optional[transforms.Compose]
            Torchvision transforms to apply to each image sample.
        """
        self.samples = samples
        self.transform = transform or get_default_transforms(is_train=False)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.samples[idx]
        img_path = Path(item["image_path"])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as err:
            logger.error(f"Error loading image {img_path}: {err}. Using blank tensor fallback.")
            image = Image.new("RGB", config.SPECTROGRAM_SHAPE, color=0)

        if self.transform is not None:
            image_tensor = self.transform(image)

        label_tensor = torch.tensor(item["class_id"], dtype=torch.long)
        return image_tensor, label_tensor


def create_pytorch_dataloaders(
    manifest_path: Optional[Union[str, Path]] = None,
    batch_size: int = config.TRAINING.batch_size,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = config.TRAINING.seed,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Create train, validation, and test PyTorch DataLoaders from manifest data.

    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]
        (train_loader, val_loader, test_loader, split_counts)
    """
    if manifest_path is None:
        manifest_path = config.PROCESSED_DATA_DIR / config.PROCESSED_MANIFEST_NAME

    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Manifest file not found at {path}. Run signal conversion first."
        )

    with open(path, "r") as f:
        samples: List[Dict[str, Union[str, int]]] = json.load(f)

    if not samples:
        raise ValueError(f"Manifest file at {path} is empty.")

    # Stratified shuffle & split by class
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(samples), generator=rng).tolist()
    shuffled_samples = [samples[i] for i in indices]

    n_total = len(shuffled_samples)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_samples = shuffled_samples[:n_train]
    val_samples = shuffled_samples[n_train : n_train + n_val]
    test_samples = shuffled_samples[n_train + n_val :]

    train_dataset = RadarSpectrogramDataset(
        train_samples, transform=get_default_transforms(is_train=True)
    )
    val_dataset = RadarSpectrogramDataset(
        val_samples, transform=get_default_transforms(is_train=False)
    )
    test_dataset = RadarSpectrogramDataset(
        test_samples, transform=get_default_transforms(is_train=False)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    split_counts = {
        "train": len(train_samples),
        "val": len(val_samples),
        "test": len(test_samples),
        "total": n_total,
    }

    logger.info(f"DataLoaders created successfully. Split summary: {split_counts}")
    return train_loader, val_loader, test_loader, split_counts
