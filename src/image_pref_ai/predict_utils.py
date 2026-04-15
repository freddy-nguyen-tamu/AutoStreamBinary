from pathlib import Path

import torch
from PIL import Image

from .config import MODEL_PATH
from .data import build_transform
from .model import build_model, get_device


def load_trained_model(model_path: Path = MODEL_PATH):
    device = get_device()

    checkpoint = torch.load(model_path, map_location=device)

    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, device


@torch.no_grad()
def predict_image(image_path: Path, model, device) -> tuple[int, float]:
    transform = build_transform(train=False)

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    logits = model(image_tensor)
    probabilities = torch.softmax(logits, dim=1)[0]

    confidence_for_1 = float(probabilities[1].detach().cpu().item())
    prediction = int(torch.argmax(probabilities).detach().cpu().item())

    return prediction, confidence_for_1
