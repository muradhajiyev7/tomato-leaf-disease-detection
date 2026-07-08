import torch


def make_tta_batch(image_tensor: torch.Tensor) -> torch.Tensor:
    """Create a small TTA batch from a single preprocessed image tensor."""

    if image_tensor.ndim != 4 or image_tensor.size(0) != 1:
        raise ValueError("Expected image tensor shape [1, C, H, W].")

    original = image_tensor
    hflip = torch.flip(image_tensor, dims=[3])
    vflip = torch.flip(image_tensor, dims=[2])
    return torch.cat([original, hflip, vflip], dim=0)


@torch.no_grad()
def predict_with_tta(model, image_tensor: torch.Tensor) -> torch.Tensor:
    batch = make_tta_batch(image_tensor)
    logits = model(batch)
    return logits.mean(dim=0, keepdim=True)

