from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "dataset"

# Put the images for the current training run here:
# dataset/current/0 = discard
# dataset/current/1 = preferred
CURRENT_TRAIN_DIR = DATASET_DIR / "current"

# Optional validation folders:
# dataset/val/0
# dataset/val/1
VAL_DIR = DATASET_DIR / "val"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "image_preference_model.pt"
LATEST_MODEL_PATH = MODELS_DIR / "image_preference_model.latest.pt"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 5
DEFAULT_LEARNING_RATE = 5e-5

# These preserve old model behavior during incremental training.
DEFAULT_DISTILL_WEIGHT = 0.5
DEFAULT_WEIGHT_DRIFT_WEIGHT = 1e-4
DEFAULT_DISTILL_TEMPERATURE = 2.0

NUM_WORKERS = 0

CLASS_NAMES = ["0", "1"]