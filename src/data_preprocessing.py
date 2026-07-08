from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image
from torchvision import transforms

from src.config import IMAGE_SIZE


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_image(image_path: str | Path) -> Image.Image:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def get_inference_transform(image_size: int = IMAGE_SIZE) -> Callable:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def preprocess_image(image_path: str | Path, image_size: int = IMAGE_SIZE):
    image = load_image(image_path)
    transform = get_inference_transform(image_size)
    return transform(image).unsqueeze(0)


def to_numpy_image(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"))


def get_training_transform(image_size: int = IMAGE_SIZE) -> Callable:
    # Keep this lightweight for the public repo. The original heavy training ran on HPC.
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_validation_transform(image_size: int = IMAGE_SIZE) -> Callable:
    return get_inference_transform(image_size)

