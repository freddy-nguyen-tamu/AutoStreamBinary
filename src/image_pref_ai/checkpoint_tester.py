import csv
import shutil
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import MODEL_PATH, NUM_WORKERS, PROJECT_ROOT
from .data import ImagePreferenceDataset
from .model import build_model, get_device

VALID_SELECTION_METRICS = {
    "accuracy",
    "balanced_accuracy",
    "precision_for_1",
    "recall_for_1",
    "f1_for_1",
}


def find_checkpoints(checkpoints_dir: Path) -> list[Path]:
    return sorted(checkpoints_dir.glob("*.pt"))


def load_checkpoint(path: Path, device):
    checkpoint = torch.load(path, map_location=device)
    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    epoch = checkpoint.get("epoch", "unknown")
    best_val_acc = checkpoint.get("best_val_acc", None)
    return model, epoch, best_val_acc


@torch.no_grad()
def evaluate_checkpoint(model, dataloader, device):
    all_targets = []
    all_predictions = []
    all_confidence_for_1 = []

    for images, labels in tqdm(dataloader, desc="Testing", leave=False):
        images = images.to(device)
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

        all_targets.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())
        all_confidence_for_1.extend(probabilities[:, 1].cpu().tolist())

    accuracy = accuracy_score(all_targets, all_predictions)
    balanced_accuracy = balanced_accuracy_score(all_targets, all_predictions)
    precision = precision_score(
        all_targets,
        all_predictions,
        zero_division=0,
    )
    recall = recall_score(
        all_targets,
        all_predictions,
        zero_division=0,
    )
    f1 = f1_score(
        all_targets,
        all_predictions,
        zero_division=0,
    )
    avg_confidence_for_1 = sum(all_confidence_for_1) / len(all_confidence_for_1)

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision_for_1": precision,
        "recall_for_1": recall,
        "f1_for_1": f1,
        "avg_confidence_for_1": avg_confidence_for_1,
        "targets": all_targets,
        "predictions": all_predictions,
    }


def test_checkpoints_and_select_best(
    test_dir: Path,
    checkpoints_dir: Path,
    target_model_path: Path = MODEL_PATH,
    output_csv: Path | None = None,
    batch_size: int = 32,
    selection_metric: str = "balanced_accuracy",
    detailed: bool = False,
    select_best: bool = True,
):
    if selection_metric not in VALID_SELECTION_METRICS:
        raise ValueError(
            f"Invalid selection metric: {selection_metric}. "
            f"Valid choices: {sorted(VALID_SELECTION_METRICS)}"
        )

    if output_csv is None:
        output_csv = PROJECT_ROOT / "outputs" / "checkpoint_test_results.csv"

    print("\n" + "=" * 80)
    print("Auto-testing epoch checkpoints")
    print("=" * 80)
    print(f"Test folder: {test_dir}")

    try:
        test_dataset = ImagePreferenceDataset(test_dir, train=False)
    except RuntimeError:
        print("No validation/test images found. Skipping checkpoint auto-test.")
        print("Put test images in dataset/val/0 and dataset/val/1 to enable it.")
        return None

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    print(f"Test images: {len(test_dataset)}")

    checkpoint_paths = find_checkpoints(checkpoints_dir)
    if not checkpoint_paths:
        print(f"No epoch checkpoints found in: {checkpoints_dir}")
        print("Skipping checkpoint auto-test.")
        return None

    print(f"Checkpoints to test: {len(checkpoint_paths)}")

    device = get_device()
    print(f"Using device: {device}")

    results = []

    for checkpoint_path in checkpoint_paths:
        print("\n" + "-" * 80)
        print(f"Testing checkpoint: {checkpoint_path}")

        model, epoch, previous_best_val_acc = load_checkpoint(checkpoint_path, device)
        metrics = evaluate_checkpoint(
            model=model,
            dataloader=test_loader,
            device=device,
        )

        row = {
            "checkpoint": str(checkpoint_path),
            "epoch": epoch,
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "precision_for_1": metrics["precision_for_1"],
            "recall_for_1": metrics["recall_for_1"],
            "f1_for_1": metrics["f1_for_1"],
            "avg_confidence_for_1": metrics["avg_confidence_for_1"],
            "previous_best_val_acc": previous_best_val_acc,
        }
        results.append(row)

        print(f"Epoch:             {epoch}")
        print(f"Accuracy:          {metrics['accuracy']:.4f}")
        print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"Precision for 1:   {metrics['precision_for_1']:.4f}")
        print(f"Recall for 1:      {metrics['recall_for_1']:.4f}")
        print(f"F1 for 1:          {metrics['f1_for_1']:.4f}")
        print(f"Avg conf for 1:    {metrics['avg_confidence_for_1']:.4f}")

        if detailed:
            print("\nClassification report:")
            print(
                classification_report(
                    metrics["targets"],
                    metrics["predictions"],
                    target_names=["0", "1"],
                    zero_division=0,
                )
            )
            print("Confusion matrix:")
            print(confusion_matrix(metrics["targets"], metrics["predictions"]))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "checkpoint",
                "epoch",
                "accuracy",
                "balanced_accuracy",
                "precision_for_1",
                "recall_for_1",
                "f1_for_1",
                "avg_confidence_for_1",
                "previous_best_val_acc",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    best_row = max(results, key=lambda row: row[selection_metric])
    best_checkpoint = Path(best_row["checkpoint"])

    print("\n" + "=" * 80)
    print("Best checkpoint after this training batch")
    print("=" * 80)
    print(f"Selection metric: {selection_metric}")
    print(f"Best checkpoint:  {best_checkpoint}")
    print(f"Epoch:            {best_row['epoch']}")
    print(f"Score:            {best_row[selection_metric]:.4f}")
    print(f"Saved results to: {output_csv}")

    if select_best:
        target_model_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path = target_model_path.with_suffix(".before_auto_select.pt")

        if target_model_path.exists():
            shutil.copy2(target_model_path, backup_path)
            print(f"Backed up previous main model to: {backup_path}")

        shutil.copy2(best_checkpoint, target_model_path)
        print("\nSelected best checkpoint as main model.")
        print(f"Copied best checkpoint to: {target_model_path}")

    return best_row
