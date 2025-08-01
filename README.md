VirtualStain-Zoo
=================

CycleGan and Pix2Pix models: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

This repository contains a minimal training setup using [Hydra](https://github.com/facebookresearch/hydra).

## Usage

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Generate a small dummy dataset:

```bash
python scripts/generate_dummy_data.py --out_dir dummy_vs_dataset --n 48 --size 256
```

Run training by specifying a config file with `--config-name`:

```bash
# CycleGAN example
python train.py --config-name=config_CGAN

# Pix2Pix example
python train.py --config-name=config_pix2pix

# UNet example
python train.py --config-name=config_unet
```

Each configuration file lives in the `configs/` directory.
