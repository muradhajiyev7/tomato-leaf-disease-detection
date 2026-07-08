from pathlib import Path

import pytest
from PIL import Image

from src.data_preprocessing import load_image


def test_load_image_returns_rgb(tmp_path: Path):
    image_path = tmp_path / "leaf.jpg"
    Image.new("RGB", (16, 16), color=(20, 120, 40)).save(image_path)

    image = load_image(image_path)

    assert image.mode == "RGB"
    assert image.size == (16, 16)


def test_load_image_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "missing.jpg")
