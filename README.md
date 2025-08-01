# VirtualStain-Zoo

A small PyTorch Lightning project for experimenting with virtual staining models. It includes implementations of **UNet**, **Pix2Pix** and **CycleGAN** along with a minimal data module.

## Project Goals

The goal is to provide simple yet complete examples of deep-learning based virtual staining. Models can transform "unstained" microscopy images into a stained appearance and serve as a baseline for research or teaching.

## Dataset Structure

`PairFolderDataModule` expects the following directory layout:

```
<dataset_root>/
  ├─ unstained/
  │    └─ 000.png
  └─ stained/
       └─ 000.png
```

Files must have matching names across folders. The datamodule performs an 80/20 train/validation split automatically.

A small synthetic dataset can be generated with:

```
python scripts/generate_dummy_data.py --out_dir data/dummy --n 48 --size 256
```

## Installation

Install the required packages using `requirements.txt`:

```
pip install -r requirements.txt
```

PyTorch is installed with CPU support by default; feel free to install a GPU build if needed.

## Training with Hydra

Training uses Hydra configs stored in `configs/`. Specify a config file and optionally override parameters:

```
python train.py --config-name=config_pix2pix dataset.data_root=/path/to/data

# Override values on the command line
python train.py --config-name=config_unet trainer.max_epochs=10 model.lr=0.0001
```

Running `python train.py` without arguments uses `config_cGAN.yaml`.

## Model Options

Models are registered in the `models/` package and selected via `model.name` in the config:

- `UNetVS` – small encoder/decoder baseline.
- `Pix2Pix` – paired training using a UNet generator with a PatchGAN discriminator.
- `CycleGAN` – unpaired translation with dual generators and discriminators.

