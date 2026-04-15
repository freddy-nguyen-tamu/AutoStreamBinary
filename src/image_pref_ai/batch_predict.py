import argparse
import csv
from pathlib import Path

from tqdm import tqdm

from .config import MODEL_PATH
from .data import list_images
from .predict_utils import load_trained_model, predict_image


def parse_args():
    parser = argparse.ArgumentParser(description="Predict 0 or 1 for a folder of images.")
    parser.add_argument("--input_folder", required=True, help="Folder containing images.")
    parser.add_argument(
        "--output_csv",
        default="outputs/predictions.csv",
        help="Where to save prediction CSV.",
    )
    parser.add_argument(
        "--model",
        default=str(MODEL_PATH),
        help="Path to trained model .pt file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_folder = Path(args.input_folder)
    output_csv = Path(args.output_csv)
    model_path = Path(args.model)

    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Run python train.py first."
        )

    image_paths = list_images(input_folder)
    if not image_paths:
        raise RuntimeError(f"No supported images found in: {input_folder}")

    model, device = load_trained_model(model_path)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "prediction", "confidence_for_1"])

        for image_path in tqdm(image_paths, desc="Predicting"):
            try:
                prediction, confidence_for_1 = predict_image(image_path, model, device)
                writer.writerow([str(image_path), prediction, f"{confidence_for_1:.6f}"])
            except Exception as exc:
                writer.writerow([str(image_path), "ERROR", str(exc)])

    print(f"Saved predictions to: {output_csv}")


if __name__ == "__main__":
    main()
