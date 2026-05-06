import argparse
import csv
import shutil
from pathlib import Path

from tqdm import tqdm

from .config import MODEL_PATH
from .data import list_images
from .predict_utils import load_trained_model, predict_image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict images in a folder and sort them into keep/discard folders."
    )

    parser.add_argument(
        "--input_folder",
        required=True,
        help="Folder containing images to sort.",
    )

    parser.add_argument(
        "--model",
        default=str(MODEL_PATH),
        help="Path to trained model .pt file.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for keep. Default: 0.5",
    )

    parser.add_argument(
        "--mode",
        choices=["copy", "move"],
        default="copy",
        help="copy keeps originals, move relocates originals. Default: copy",
    )

    parser.add_argument(
        "--output_csv",
        default=None,
        help="Optional CSV path. Default: input_folder/predictions.csv",
    )

    parser.add_argument(
        "--include_subfolders",
        action="store_true",
        help="Also sort images inside subfolders.",
    )

    return parser.parse_args()


def unique_destination_path(destination_folder: Path, source_path: Path) -> Path:
    destination_path = destination_folder / source_path.name

    if not destination_path.exists():
        return destination_path

    counter = 1

    while True:
        new_name = f"{source_path.stem}_{counter}{source_path.suffix}"
        destination_path = destination_folder / new_name

        if not destination_path.exists():
            return destination_path

        counter += 1


def list_top_level_images(folder: Path) -> list[Path]:
    from .config import IMAGE_EXTENSIONS

    paths = []

    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(path)

    return sorted(paths)


def main():
    args = parse_args()

    input_folder = Path(args.input_folder)
    model_path = Path(args.model)

    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    if not input_folder.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {input_folder}")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Run python train.py first."
        )

    keep_folder = input_folder / "keep"
    discard_folder = input_folder / "discard"

    keep_folder.mkdir(parents=True, exist_ok=True)
    discard_folder.mkdir(parents=True, exist_ok=True)

    if args.include_subfolders:
        image_paths = list_images(input_folder)

        # Do not re-process files already inside output folders.
        image_paths = [
            path
            for path in image_paths
            if keep_folder not in path.parents and discard_folder not in path.parents
        ]
    else:
        image_paths = list_top_level_images(input_folder)

    if not image_paths:
        raise RuntimeError(f"No supported images found in: {input_folder}")

    if args.output_csv is None:
        output_csv = input_folder / "predictions.csv"
    else:
        output_csv = Path(args.output_csv)

    model, device = load_trained_model(model_path)

    rows = []

    print(f"Input folder: {input_folder}")
    print(f"Images found: {len(image_paths)}")
    print(f"Keep folder: {keep_folder}")
    print(f"Discard folder: {discard_folder}")
    print(f"Mode: {args.mode}")
    print(f"Keep threshold: {args.threshold}")

    for image_path in tqdm(image_paths, desc="Predicting and sorting"):
        try:
            prediction, confidence_for_1 = predict_image(image_path, model, device)

            if confidence_for_1 >= args.threshold:
                final_label = "keep"
                destination_folder = keep_folder
            else:
                final_label = "discard"
                destination_folder = discard_folder

            destination_path = unique_destination_path(destination_folder, image_path)

            if args.mode == "move":
                shutil.move(str(image_path), str(destination_path))
            else:
                shutil.copy2(image_path, destination_path)

            rows.append(
                {
                    "source_path": str(image_path),
                    "destination_path": str(destination_path),
                    "prediction": prediction,
                    "confidence_for_1": f"{confidence_for_1:.6f}",
                    "final_folder": final_label,
                    "mode": args.mode,
                }
            )

        except Exception as exc:
            rows.append(
                {
                    "source_path": str(image_path),
                    "destination_path": "",
                    "prediction": "ERROR",
                    "confidence_for_1": "",
                    "final_folder": "",
                    "mode": args.mode,
                    "error": str(exc),
                }
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "source_path",
        "destination_path",
        "prediction",
        "confidence_for_1",
        "final_folder",
        "mode",
        "error",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            if "error" not in row:
                row["error"] = ""
            writer.writerow(row)

    keep_count = sum(1 for row in rows if row.get("final_folder") == "keep")
    discard_count = sum(1 for row in rows if row.get("final_folder") == "discard")
    error_count = sum(1 for row in rows if row.get("prediction") == "ERROR")

    print("\nDone.")
    print(f"Kept: {keep_count}")
    print(f"Discarded: {discard_count}")
    print(f"Errors: {error_count}")
    print(f"CSV saved to: {output_csv}")


if __name__ == "__main__":
    main()
