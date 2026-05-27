# AutoStreamBinary Image Preference Model

This project trains a binary image preference classifier.

The model predicts:

- `0`: discard
- `1`: preferred

It supports incremental training with two old-memory retention methods:

- teacher distillation from the previous saved model
- an automatic replay buffer from older examples

The default daily command is:

```bash
python train.py
```

By default, that runs up to 30 epochs, creates validation images if needed, uses replay, saves every epoch checkpoint, early-stops on balanced accuracy, auto-tests checkpoints, and selects the best checkpoint as the main model.

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
  checkpoints/
    epoch_XXXX.pt

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

By default, training:

- creates `dataset/val/0` and `dataset/val/1` from `dataset/current` if validation is empty
- trains on `dataset/current + dataset/replay`
- validates after every epoch
- saves every epoch to `models/checkpoints/epoch_XXXX.pt`
- early-stops on `balanced_accuracy` with patience `3`
- tests all checkpoints after training
- copies the best checkpoint by `balanced_accuracy` to `models/image_preference_model.pt`
- copies a random sample of current images into `dataset/replay` for future runs

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

Disable early stopping:

```bash
python train.py --no_early_stopping
```

Disable checkpoint auto-testing and best-epoch selection:

```bash
python train.py --no_auto_test
```

Train fewer epochs manually:

```bash
python train.py --epochs 5
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

Validation images are created automatically from `dataset/current` if `dataset/val/0` or `dataset/val/1` is empty.

You can also put validation images there yourself:

```text
dataset/val/0
dataset/val/1
```

Training prints validation loss, validation accuracy, balanced accuracy, a final classification report, and a confusion matrix.

If validation folders have no supported images, training continues without validation.

## Saved Models

Training saves:

```text
models/image_preference_model.pt
models/image_preference_model.latest.pt
models/checkpoints/epoch_XXXX.pt
```

The main model file is overwritten during training, then automatic checkpoint testing copies the best checkpoint back to `models/image_preference_model.pt`.

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
epochs:                  30
learning rate:           5e-5
distill weight:          0.5
weight drift weight:     1e-4
distill temperature:     2.0
replay fraction:         0.20
replay max per class:    500
replay seed:             42
auto val fraction:       0.20
auto val max per class:  200
auto val seed:           123
early stopping:          enabled
early stopping metric:   balanced_accuracy
early stopping patience: 3
min delta:               0.0001
auto checkpoint testing: enabled
selection metric:        balanced_accuracy
```
