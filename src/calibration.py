import torch
import torch.nn as nn


class TemperatureScaler(nn.Module):
    """Simple temperature scaling wrapper for calibrated confidence scores."""

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(float(temperature)))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temperature = torch.clamp(self.temperature, min=1e-6)
        return logits / temperature


def apply_temperature_scaling(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    temperature = max(float(temperature), 1e-6)
    return logits / temperature

