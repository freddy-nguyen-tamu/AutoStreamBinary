import argparse
from pathlib import Path

from .config import MODEL_PATH
from .predict_utils import load_trained_model, predict_image


def parse_args():
    parser = argparse.ArgumentParser(description="Predict 0 or 1 for one image.")
    parser.add_argument("--image", required=True, help="Path to image.")
    parser.add_argument(
        "--model",
        default=str(MODEL_PATH),
        help="Path to trained model .pt file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_path = Path(args.image)
    model_path = Path(args.model)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Run python train.py first."
        )

    model, device = load_trained_model(model_path)
    prediction, confidence_for_1 = predict_image(image_path, model, device)

    print(f"Image: {image_path}")
    print(f"Prediction: {prediction}")
    print(f"Confidence for 1: {confidence_for_1:.4f}")


if __name__ == "__main__":
    main()
