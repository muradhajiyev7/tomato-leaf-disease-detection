from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset

from src.config import CLASS_NAMES


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class TomatoLeafDataset(Dataset):
    """ImageFolder-style dataset with clear errors for missing local data."""

    def __init__(
        self,
        root_dir: str | Path,
        transform: Optional[Callable] = None,
        class_names: Optional[list[str]] = None,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.class_names = class_names or CLASS_NAMES
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dataset folder not found: {self.root_dir}. "
                "Place data under data/raw/Universal_Tomato_Dataset/."
            )

        self.samples = self._collect_samples()
        if not self.samples:
            raise ValueError(f"No images found under {self.root_dir}.")

    def _collect_samples(self) -> list[tuple[Path, int]]:
        samples: list[tuple[Path, int]] = []
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            for path in sorted(class_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    samples.append((path, self.class_to_idx[class_name]))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

