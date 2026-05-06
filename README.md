# AutoStreamBinary Image Preference Model

This project trains a binary image preference classifier.

The model predicts:

- `0`: discard
- `1`: preferred

It supports incremental training with two old-memory retention methods:

- teacher distillation from the previous saved model
- an automatic replay buffer from older examples

## Folder Layout

```text
dataset/
  current/
    0/    today/new discard training images
    1/    today/new preferred training images
  replay/
    0/    automatically saved old discard examples
    1/    automatically saved old preferred examples
  val/
    0/    optional validation discard images
    1/    optional validation preferred images

models/
  image_preference_model.pt
  image_preference_model.latest.pt

outputs/
  optional CSV outputs

src/image_pref_ai/
  train.py
  data.py
  model.py
  predict.py
  batch_predict.py
  sort_predict.py
  predict_utils.py
  config.py
```

Root scripts are thin launchers:

- `python train.py`
- `python predict.py`
- `python batch_predict.py`

## Install

```bash
pip install -r requirements.txt
```

## Train

Put new training images in:

```text
dataset/current/0
dataset/current/1
```

Then run:

```bash
python train.py
```

By default, training uses:

```text
dataset/current + dataset/replay
```

After training finishes, it copies a random sample of the current images into `dataset/replay` for future runs.

Default replay behavior:

- copies `20%` of current images per class
- keeps at most `500` replay images per class
- uses replay seed `42`

## Useful Training Commands

Train normally with replay enabled:

```bash
python train.py
```

Train without replay:

```bash
python train.py --no_replay
```

Copy more current images into replay after training:

```bash
python train.py --replay_fraction 0.5
```

Keep more replay examples per class:

```bash
python train.py --replay_max_per_class 1000
```

Start from pretrained ResNet18 instead of loading the saved model:

```bash
python train.py --fresh
```

Recommended replay command:

```bash
python train.py --replay_fraction 0.2 --replay_max_per_class 500
```

## Validation

Validation images are optional.

If present, put them in:

```text
dataset/val/0
dataset/val/1
```

Training prints validation loss, validation accuracy, a final classification report, and a confusion matrix.

If validation folders have no supported images, training continues without validation.

## Saved Models

Training saves:

```text
models/image_preference_model.pt
models/image_preference_model.latest.pt
```

The main model file is overwritten after each epoch.

## Predict One Image

```bash
python predict.py --image "path/to/image.jpg"
```

Use a custom model:

```bash
python predict.py --image "path/to/image.jpg" --model "models/image_preference_model.pt"
```

## Sort A Folder

```bash
python batch_predict.py --input_folder "path/to/images"
```

This creates two folders inside the input folder:

```text
path/to/images/keep
path/to/images/discard
```

By default, images are copied into `keep` or `discard`. Originals stay in place.

The sorting rule is:

```text
confidence_for_1 >= threshold  -> keep
confidence_for_1 < threshold   -> discard
```

Default threshold is `0.5`.

Move files instead of copying:

```bash
python batch_predict.py --input_folder "path/to/images" --mode move
```

Use a stricter keep threshold:

```bash
python batch_predict.py --input_folder "path/to/images" --threshold 0.8
```

Include images inside subfolders:

```bash
python batch_predict.py --input_folder "path/to/images" --include_subfolders
```

Save the CSV somewhere specific:

```bash
python batch_predict.py --input_folder "path/to/images" --output_csv "outputs/predictions.csv"
```

If `--output_csv` is not provided, the CSV is saved at:

```text
path/to/images/predictions.csv
```

The CSV contains:

```text
source_path,destination_path,prediction,confidence_for_1,final_folder,mode,error
```

## Supported Images

Supported extensions:

```text
.jpg .jpeg .png .webp .bmp
```

## Main Defaults

Defaults are defined in `src/image_pref_ai/config.py`:

```text
image size:              224
batch size:              32
epochs:                  5
learning rate:           5e-5
distill weight:          0.5
weight drift weight:     1e-4
distill temperature:     2.0
replay fraction:         0.20
replay max per class:    500
replay seed:             42
```
