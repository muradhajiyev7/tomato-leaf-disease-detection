from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from src.config import CLASS_NAMES, DEVICE, MODEL_PATH


def build_model(num_classes: int = len(CLASS_NAMES), model_name: str = "tf_efficientnet_b3"):
    try:
        import timm
    except ImportError as exc:
        raise ImportError(
            "The timm package is required for EfficientNet backbones. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    return timm.create_model(model_name, pretrained=False, num_classes=num_classes)


def load_model(
    weights_path: str | Path = MODEL_PATH,
    num_classes: int = len(CLASS_NAMES),
    device: torch.device = DEVICE,
    model_name: str = "tf_efficientnet_b3",
) -> nn.Module:
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            "Model weights are not included in this public repository. "
            "Please place weights under models/ or update MODEL_PATH in src/config.py."
        )

    model = build_model(num_classes=num_classes, model_name=model_name)
    checkpoint = torch.load(weights_path, map_location=device)

    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

