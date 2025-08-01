import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import importlib

# Import models from files
from models.unet import UNetVS
from models.pix2pix import Pix2Pix
from models.cGAN import CycleGAN


@torch.no_grad()
def test_unet_forward():
    model = UNetVS({})
    x = torch.randn(1, 3, 256, 256)
    y = model.eval()(x)
    assert y.shape == x.shape


@torch.no_grad()
def test_pix2pix_forward():
    model = Pix2Pix({})
    x = torch.randn(1, 3, 256, 256)
    y = model.eval()(x)
    assert y.shape == x.shape


@torch.no_grad()
def test_cyclegan_forward():
    model = CycleGAN({})
    x = torch.randn(1, 3, 256, 256)
    y = model.eval()(x)
    assert y.shape == x.shape
