import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import (
    CURRENT_TRAIN_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DISTILL_TEMPERATURE,
    DEFAULT_DISTILL_WEIGHT,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_WEIGHT_DRIFT_WEIGHT,
    LATEST_MODEL_PATH,
    MODEL_PATH,
    NUM_WORKERS,
    VAL_DIR,
)
from .data import ImagePreferenceDataset
from .model import build_model, get_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Incrementally train image preference model with old-memory retention."
    )

    parser.add_argument("--train_dir", default=str(CURRENT_TRAIN_DIR))
    parser.add_argument("--val_dir", default=str(VAL_DIR))
    parser.add_argument("--model", default=str(MODEL_PATH))

    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE)

    parser.add_argument(
        "--distill_weight",
        type=float,
        default=DEFAULT_DISTILL_WEIGHT,
        help="How strongly to preserve previous model behavior.",
    )
    parser.add_argument(
        "--weight_drift_weight",
        type=float,
        default=DEFAULT_WEIGHT_DRIFT_WEIGHT,
        help="How strongly to prevent weights from drifting away from old checkpoint.",
    )
    parser.add_argument(
        "--distill_temperature",
        type=float,
        default=DEFAULT_DISTILL_TEMPERATURE,
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start from pretrained ResNet18 instead of loading saved model.",
    )

    return parser.parse_args()


def clone_frozen_teacher(model):
    teacher = copy.deepcopy(model)
    teacher.eval()

    for param in teacher.parameters():
        param.requires_grad = False

    return teacher


def snapshot_parameters(model):
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def weight_drift_loss(model, old_params):
    if not old_params:
        return torch.tensor(0.0, device=next(model.parameters()).device)

    total = torch.tensor(0.0, device=next(model.parameters()).device)
    count = 0

    for name, param in model.named_parameters():
        if name not in old_params:
            continue

        total = total + F.mse_loss(param, old_params[name], reduction="mean")
        count += 1

    if count == 0:
        return total

    return total / count


def distillation_loss(student_logits, teacher_logits, temperature):
    t = temperature

    student_log_probs = F.log_softmax(student_logits / t, dim=1)
    teacher_probs = F.softmax(teacher_logits / t, dim=1)

    return F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean",
    ) * (t * t)


def train_one_epoch(
    model,
    teacher_model,
    old_params,
    dataloader,
    criterion,
    optimizer,
    device,
    distill_weight,
    weight_drift_weight,
    distill_temperature,
):
    model.train()

    total_loss = 0.0
    total_ce_loss = 0.0
    total_distill_loss = 0.0
    total_drift_loss = 0.0

    predictions = []
    targets = []

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        student_logits = model(images)
        ce = criterion(student_logits, labels)

        loss = ce
        kd = torch.tensor(0.0, device=device)
        drift = torch.tensor(0.0, device=device)

        if teacher_model is not None and distill_weight > 0:
            with torch.no_grad():
                teacher_logits = teacher_model(images)

            kd = distillation_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                temperature=distill_temperature,
            )
            loss = loss + distill_weight * kd

        if old_params is not None and weight_drift_weight > 0:
            drift = weight_drift_loss(model, old_params)
            loss = loss + weight_drift_weight * drift

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_ce_loss += ce.item() * batch_size
        total_distill_loss += kd.item() * batch_size
        total_drift_loss += drift.item() * batch_size

        batch_predictions = torch.argmax(student_logits, dim=1)
        predictions.extend(batch_predictions.detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())

    dataset_size = len(dataloader.dataset)

    return {
        "loss": total_loss / dataset_size,
        "ce_loss": total_ce_loss / dataset_size,
        "distill_loss": total_distill_loss / dataset_size,
        "drift_loss": total_drift_loss / dataset_size,
        "accuracy": accuracy_score(targets, predictions),
    }


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    predictions = []
    targets = []

    for images, labels in tqdm(dataloader, desc="Validation", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)

        batch_predictions = torch.argmax(logits, dim=1)
        predictions.extend(batch_predictions.detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(targets, predictions)

    return avg_loss, accuracy, targets, predictions


def load_checkpoint_if_available(model, optimizer, model_path, device, fresh):
    if fresh:
        print("Fresh training requested. Ignoring saved model.")
        return 0, -1.0, False

    if not model_path.exists():
        print("No saved model found. Starting from pretrained ResNet18.")
        return 0, -1.0, False

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except ValueError:
            print("Could not load optimizer state. Continuing with fresh optimizer.")

    previous_epoch = int(checkpoint.get("epoch", 0))
    best_val_acc = float(checkpoint.get("best_val_acc", -1.0))

    print(f"Loaded saved model: {model_path}")
    print(f"Previous epoch: {previous_epoch}")

    if best_val_acc >= 0:
        print(f"Previous best validation accuracy: {best_val_acc:.4f}")

    return previous_epoch, best_val_acc, True


def save_checkpoint(model, optimizer, epoch, best_val_acc, model_path):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    LATEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_acc": best_val_acc,
    }

    torch.save(payload, model_path)
    torch.save(payload, LATEST_MODEL_PATH)


def main():
    args = parse_args()

    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir)
    model_path = Path(args.model)

    print(f"Training folder: {train_dir}")

    train_dataset = ImagePreferenceDataset(train_dir, train=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    print(f"Training images this run: {len(train_dataset)}")

    val_loader = None

    try:
        val_dataset = ImagePreferenceDataset(val_dir, train=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )
        print(f"Validation images: {len(val_dataset)}")
    except RuntimeError:
        print("No validation images found. Training without validation.")

    device = get_device()
    print(f"Using device: {device}")

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    previous_epoch, best_val_acc, resumed = load_checkpoint_if_available(
        model=model,
        optimizer=optimizer,
        model_path=model_path,
        device=device,
        fresh=args.fresh,
    )

    teacher_model = None
    old_params = None

    if resumed:
        teacher_model = clone_frozen_teacher(model).to(device)
        old_params = snapshot_parameters(model)

        print("Old-memory retention is active.")
        print(f"Distillation weight: {args.distill_weight}")
        print(f"Weight-drift weight: {args.weight_drift_weight}")
    else:
        print("Old-memory retention is inactive because this is the first/fresh run.")

    final_targets = []
    final_predictions = []

    for local_epoch in range(1, args.epochs + 1):
        global_epoch = previous_epoch + local_epoch

        print(f"\nEpoch this run: {local_epoch}/{args.epochs}")
        print(f"Total continued epoch: {global_epoch}")

        metrics = train_one_epoch(
            model=model,
            teacher_model=teacher_model,
            old_params=old_params,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            distill_weight=args.distill_weight,
            weight_drift_weight=args.weight_drift_weight,
            distill_temperature=args.distill_temperature,
        )

        print(
            "Train "
            f"loss: {metrics['loss']:.4f} | "
            f"acc: {metrics['accuracy']:.4f} | "
            f"ce: {metrics['ce_loss']:.4f} | "
            f"distill: {metrics['distill_loss']:.4f} | "
            f"drift: {metrics['drift_loss']:.6f}"
        )

        if val_loader is not None:
            val_loss, val_acc, targets, predictions = evaluate(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
            )

            final_targets = targets
            final_predictions = predictions

            print(f"Val loss:   {val_loss:.4f} | Val acc:   {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print("New best validation accuracy.")

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=global_epoch,
            best_val_acc=best_val_acc,
            model_path=model_path,
        )

        print(f"Saved model checkpoint: {model_path}")

    if val_loader is not None and final_targets and final_predictions:
        print("\nFinal validation report:")
        print(classification_report(final_targets, final_predictions, target_names=["0", "1"]))

        print("Confusion matrix:")
        print(confusion_matrix(final_targets, final_predictions))

    print("\nDone.")
    print(f"Model saved at: {model_path}")


if __name__ == "__main__":
    main()
