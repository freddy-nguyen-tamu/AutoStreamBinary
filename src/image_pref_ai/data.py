from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .config import IMAGE_EXTENSIONS, IMAGE_SIZE


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []

    paths: list[Path] = []
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(path)

    return sorted(paths)


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


def folder_samples(root: Path) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []

    for label in [0, 1]:
        class_dir = root / str(label)
        for image_path in list_images(class_dir):
            samples.append((image_path, label))

    return sorted(samples, key=lambda item: str(item[0]))


class ImagePreferenceDataset(Dataset):
    def __init__(self, root: Path, train: bool) -> None:
        self.root = Path(root)
        self.samples = folder_samples(self.root)

        if not self.samples:
            raise RuntimeError(
                f"No images found in {self.root}. "
                "Expected folders named 0 and 1 containing images."
            )

        self.transform = build_transform(train=train)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to load image: {image_path}") from exc

        image = self.transform(image)
        return image, label
