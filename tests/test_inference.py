from src.inference import should_accept_prediction, summarize_probabilities
import pytest


def test_should_accept_prediction_requires_confidence_and_margin():
    assert should_accept_prediction(0.90, 0.20) is True
    assert should_accept_prediction(0.70, 0.20) is False
    assert should_accept_prediction(0.90, 0.01) is False


def test_summarize_probabilities_returns_top_class():
    import torch

    probabilities = torch.tensor([0.10, 0.80, 0.10])
    predicted_idx, confidence, margin = summarize_probabilities(probabilities)

    assert predicted_idx == 1
    assert confidence == pytest.approx(0.80)
    assert margin == pytest.approx(0.70)
