import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_PATH
from src.inference import predict_image


MISSING_WEIGHTS_MESSAGE = (
    "Model weights are not included in this public repository. "
    "Please place weights under models/ or update MODEL_PATH in src/config.py."
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run tomato leaf disease inference.")
    parser.add_argument("--image", required=True, help="Path to a tomato leaf image.")
    parser.add_argument("--weights", default=str(MODEL_PATH), help="Path to model weights.")
    parser.add_argument("--no-tta", action="store_true", help="Disable test-time augmentation.")
    return parser.parse_args()


def main():
    args = parse_args()
    weights_path = Path(args.weights)

    if not weights_path.exists():
        print(MISSING_WEIGHTS_MESSAGE)
        return 0

    result = predict_image(args.image, weights_path=weights_path, use_tta=not args.no_tta)
    status = "accepted" if result.accepted else "rejected"

    print(f"Predicted disease: {result.predicted_class}")
    print(f"Confidence       : {result.confidence:.2%}")
    print(f"Margin           : {result.margin:.2%}")
    print(f"Decision         : {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
