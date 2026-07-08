from dataclasses import dataclass
from pathlib import Path

import torch

from src.calibration import apply_temperature_scaling
from src.config import (
    ACCEPT_CONFIDENCE_THRESHOLD,
    CLASS_NAMES,
    DEVICE,
    MODEL_PATH,
    TEMPERATURE,
    UNCERTAINTY_MARGIN_THRESHOLD,
)
from src.data_preprocessing import preprocess_image
from src.model import load_model
from src.tta import predict_with_tta


@dataclass
class PredictionResult:
    predicted_class: str
    confidence: float
    accepted: bool
    margin: float


def should_accept_prediction(
    confidence: float,
    margin: float,
    confidence_threshold: float = ACCEPT_CONFIDENCE_THRESHOLD,
    margin_threshold: float = UNCERTAINTY_MARGIN_THRESHOLD,
) -> bool:
    return confidence >= confidence_threshold and margin >= margin_threshold


def summarize_probabilities(probabilities: torch.Tensor) -> tuple[int, float, float]:
    top_values, top_indices = torch.topk(probabilities, k=min(2, probabilities.numel()))
    confidence = float(top_values[0].item())
    predicted_idx = int(top_indices[0].item())
    second_confidence = float(top_values[1].item()) if top_values.numel() > 1 else 0.0
    margin = confidence - second_confidence
    return predicted_idx, confidence, margin


@torch.no_grad()
def predict_image(
    image_path: str | Path,
    weights_path: str | Path = MODEL_PATH,
    use_tta: bool = True,
    temperature: float = TEMPERATURE,
) -> PredictionResult:
    model = load_model(weights_path=weights_path, device=DEVICE)
    image_tensor = preprocess_image(image_path).to(DEVICE)

    logits = predict_with_tta(model, image_tensor) if use_tta else model(image_tensor)
    logits = apply_temperature_scaling(logits, temperature=temperature)
    probabilities = torch.softmax(logits.squeeze(0), dim=0)

    predicted_idx, confidence, margin = summarize_probabilities(probabilities)
    accepted = should_accept_prediction(confidence, margin)

    return PredictionResult(
        predicted_class=CLASS_NAMES[predicted_idx],
        confidence=confidence,
        accepted=accepted,
        margin=margin,
    )

