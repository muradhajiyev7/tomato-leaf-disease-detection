from pathlib import Path

from PIL import Image

from src.dataset import TomatoLeafDataset


def test_dataset_collects_class_images(tmp_path: Path):
    class_dir = tmp_path / "Healthy"
    class_dir.mkdir()
    Image.new("RGB", (16, 16), color=(40, 120, 40)).save(class_dir / "sample.jpg")

    dataset = TomatoLeafDataset(tmp_path, class_names=["Healthy"])

    assert len(dataset) == 1
    _, label = dataset[0]
    assert label == 0
