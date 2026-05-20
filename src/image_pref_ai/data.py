from datetime import datetime
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms

from .config import IMAGE_EXTENSIONS, IMAGE_SIZE, OUTPUTS_DIR

DATA_VALIDATION_LOG_PATH = OUTPUTS_DIR / "image_validation_log.txt"


def log_data_message(message: str) -> None:
    print(message)
    DATA_VALIDATION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with DATA_VALIDATION_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")
        f.flush()


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []

    paths = []

    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(path)

    return sorted(paths)


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def build_transform(train: bool):
    if train:
        return transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=8),
                transforms.ColorJitter(
                    brightness=0.12,
                    contrast=0.12,
                    saturation=0.12,
                    hue=0.03,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def folder_samples(root: Path, validate_images: bool = True) -> list[tuple[Path, int]]:
    samples = []
    skipped_bad = 0
    candidates = 0

    log_data_message(f"Scanning image folder: {root}")

    for label in [0, 1]:
        class_dir = root / str(label)

        for image_path in list_images(class_dir):
            candidates += 1

            if validate_images and not is_valid_image(image_path):
                skipped_bad += 1
                log_data_message(f"Skipping unreadable image: {image_path}")
                continue

            samples.append((image_path, label))

    if skipped_bad:
        log_data_message(f"Skipped {skipped_bad} unreadable image(s) from {root}")

    log_data_message(
        f"Image scan complete for {root}: "
        f"{len(samples)} valid image(s), {skipped_bad} skipped, {candidates} candidate(s)."
    )

    return sorted(samples, key=lambda item: str(item[0]))


class ImagePreferenceDataset(Dataset):
    def __init__(self, root: Path, train: bool) -> None:
        self.root = Path(root)
        self.samples = folder_samples(self.root, validate_images=True)

        if not self.samples:
            raise RuntimeError(
                f"No valid images found in {self.root}. "
                "Expected folders named 0 and 1 containing readable images."
            )

        self.transform = build_transform(train=train)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]

        try:
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)
            return image, label
        except Exception as exc:
            log_data_message(f"Failed to load image after validation: {image_path}")
            raise RuntimeError(f"Failed to load image after validation: {image_path}") from exc


class MultiFolderPreferenceDataset(Dataset):
    """
    Loads images from multiple folder roots.

    Example:
        roots = [
            dataset/current,
            dataset/replay,
        ]

    Each root should contain:
        0/
        1/
    """

    def __init__(self, roots: list[Path], train: bool) -> None:
        self.roots = [Path(root) for root in roots]
        self.samples = []

        for root in self.roots:
            self.samples.extend(folder_samples(root, validate_images=True))

        self.samples = sorted(self.samples, key=lambda item: str(item[0]))

        if not self.samples:
            roots_text = ", ".join(str(root) for root in self.roots)
            raise RuntimeError(
                f"No valid images found in any training folder: {roots_text}. "
                "Expected folders named 0 and 1 containing readable images."
            )

        self.transform = build_transform(train=train)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]

        try:
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)
            return image, label
        except Exception as exc:
            log_data_message(f"Failed to load image after validation: {image_path}")
            raise RuntimeError(f"Failed to load image after validation: {image_path}") from exc
