# Incremental Image Preference AI With Old-Memory Retention

This project trains an AI to label images as:

```text
1 = preferred / keep
0 = discard / reject
```

It supports incremental training:

```text
Run 1:
  train on batch 1
  save model

Run 2:
  remove old images yourself
  add completely new images
  run training again
  it loads the old model and continues from it
```

Unlike plain continued training, this version tries to keep the old preference in mind even when the old images are not present.

It does that with:

```text
1. Checkpoint resume
2. Frozen teacher model from the previous checkpoint
3. Knowledge-distillation loss
4. Weight-drift penalty
```

So old images still affect the model through the saved weights.

---

## Folder layout

```text
dataset/
  current/
    0/   discard images for this training run
    1/   preferred images for this training run
  val/
    0/   optional validation discard images
    1/   optional validation preferred images
```

Use `dataset/current` for the images you want to train on **right now**.

After training, move/delete those images yourself and put the next new batch there.

---

## Install

### Windows PowerShell

```powershell
cd image_preference_ai_incremental_retention
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Linux / macOS

```bash
cd image_preference_ai_incremental_retention
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Train / continue training

```bash
python train.py
```

The model is saved to:

```text
models/image_preference_model.pt
```

Every epoch also saves:

```text
models/image_preference_model.latest.pt
```

When `models/image_preference_model.pt` exists, the next training run loads it and uses it as the old-memory teacher.

---

## Useful training options

Continue training normally:

```bash
python train.py
```

Train for fewer epochs:

```bash
python train.py --epochs 3
```

Start from scratch:

```bash
python train.py --fresh
```

Increase old-memory retention:

```bash
python train.py --distill_weight 1.0 --weight_drift_weight 0.001
```

Decrease old-memory retention and adapt more aggressively to new images:

```bash
python train.py --distill_weight 0.1 --weight_drift_weight 0.00001
```

Recommended default:

```bash
python train.py --distill_weight 0.5 --weight_drift_weight 0.0001
```

---

## What the retention settings mean

### `--distill_weight`

Default:

```text
0.5
```

This makes the new model respect the previous model's behavior on the new images.

Higher value:

```text
remembers old model more
learns new batch less aggressively
```

Lower value:

```text
adapts to new batch more
may forget old preference faster
```

### `--weight_drift_weight`

Default:

```text
0.0001
```

This directly penalizes model weights for moving too far away from the previous checkpoint.

Higher value:

```text
more stable, less forgetting
```

Lower value:

```text
more flexible, more forgetting risk
```

---

## Predict one image

```bash
python predict.py --image path/to/image.jpg
```

Example:

```text
Prediction: 1
Confidence for 1: 0.8732
```

---

## Batch predict a folder

```bash
python batch_predict.py --input_folder path/to/images --output_csv outputs/predictions.csv
```

Output CSV:

```text
filepath,prediction,confidence_for_1
```

---

## Best workflow

1. Put your first batch into:

```text
dataset/current/0
dataset/current/1
```

2. Train:

```bash
python train.py
```

3. Move those images out yourself.

4. Put new images into:

```text
dataset/current/0
dataset/current/1
```

5. Train again:

```bash
python train.py
```

The old images are no longer needed in the folder, but their effect remains through the saved model checkpoint.

---

## Important note

No method can perfectly remember old images without ever seeing old examples again. The best possible method is to keep a tiny replay set of old examples. But if you do not want to do that, this project uses the next-best practical approach: old checkpoint teacher + weight-drift penalty.
