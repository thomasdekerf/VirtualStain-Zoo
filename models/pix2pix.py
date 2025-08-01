import torch
import torch.nn as nn
import torch.nn.functional as F
from core.registry import register
from core.base_model import BaseVSModel  # Adjust import if needed
import pytorch_lightning as pl

# --- Generator and Discriminator blocks ---

class UNetBlock(nn.Module):
    def __init__(self, in_c, out_c, submodule=None, outermost=False, innermost=False, use_dropout=False):
        super().__init__()
        self.outermost = outermost
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(out_c)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(in_c)
        use_bias = True

        if outermost:
            down = [nn.Conv2d(in_c, out_c, 4, 2, 1)]
            up = [
                nn.ReLU(),
                nn.ConvTranspose2d(2*out_c, in_c, 4, 2, 1),
                nn.Tanh(),
            ]
            model = down + [submodule] + up
        elif innermost:
            down = [downrelu, nn.Conv2d(in_c, out_c, 4, 2, 1)]
            up = [
                uprelu,
                nn.ConvTranspose2d(out_c, in_c, 4, 2, 1),
                upnorm
            ]
            model = down + up
        else:
            down = [downrelu, nn.Conv2d(in_c, out_c, 4, 2, 1), downnorm]
            up = [
                uprelu,
                nn.ConvTranspose2d(2*out_c, in_c, 4, 2, 1),
                upnorm
            ]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            # Add skip connection
            return torch.cat([x, self.model(x)], 1)

class UNetGenerator(nn.Module):
    def __init__(self, in_c, out_c, num_downs=8, ngf=64):
        # Build unet structure recursively
        super().__init__()
        # innermost
        unet_block = UNetBlock(ngf * 8, ngf * 8, innermost=True)
        for _ in range(num_downs - 5):
            unet_block = UNetBlock(ngf * 8, ngf * 8, submodule=unet_block, use_dropout=True)
        # next layers
        unet_block = UNetBlock(ngf * 4, ngf * 8, submodule=unet_block)
        unet_block = UNetBlock(ngf * 2, ngf * 4, submodule=unet_block)
        unet_block = UNetBlock(ngf, ngf * 2, submodule=unet_block)
        self.model = UNetBlock(in_c, ngf, submodule=unet_block, outermost=True)

    def forward(self, x):
        return self.model(x)

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_c, ndf=64, n_layers=3):
        super().__init__()
        layers = [
            nn.Conv2d(in_c, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            layers += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        layers += [
            nn.Conv2d(ndf * nf_mult, 1, 4, 1, 1)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# --- Pix2Pix Model ---

@register
class Pix2Pix(BaseVSModel):
    """Pix2Pix model for paired image-to-image virtual staining.
    Manual optimization with two optimizers (G/D), compatible with modular repo."""
    def __init__(self, hparams):
        super().__init__(hparams)
        hp = hparams if isinstance(hparams, dict) else vars(hparams)
        self.input_nc = hp.get("input_nc", 3)
        self.output_nc = hp.get("output_nc", 3)
        self.ngf = hp.get("ngf", 64)
        self.ndf = hp.get("ndf", 64)
        self.lambda_L1 = hp.get("lambda_L1", 100.0)
        self.lr = hp.get("lr", 2e-4)
        self.beta1 = hp.get("beta1", 0.5)
        self.n_layers_D = hp.get("n_layers_D", 3)
        self.automatic_optimization = False  # Needed for two opt

        self.generator = UNetGenerator(self.input_nc, self.output_nc, ngf=self.ngf)
        self.discriminator = PatchGANDiscriminator(self.input_nc + self.output_nc, ndf=self.ndf, n_layers=self.n_layers_D)
        self.loss_fn_gan = nn.BCEWithLogitsLoss()
        self.loss_fn_l1 = nn.L1Loss()

    def forward(self, x):
        return self.generator(x)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        return [opt_g, opt_d]

    def training_step(self, batch, batch_idx):
        src, tgt = batch  # src: unstained, tgt: stained
        opt_g, opt_d = self.optimizers()

        # --- 1. Train D ---
        fake_tgt = self(src).detach()
        fake_pair = torch.cat([src, fake_tgt], dim=1)
        real_pair = torch.cat([src, tgt], dim=1)
        pred_fake = self.discriminator(fake_pair)
        pred_real = self.discriminator(real_pair)
        loss_D_fake = self.loss_fn_gan(pred_fake, torch.zeros_like(pred_fake))
        loss_D_real = self.loss_fn_gan(pred_real, torch.ones_like(pred_real))
        loss_D = (loss_D_fake + loss_D_real) * 0.5

        opt_d.zero_grad()
        self.manual_backward(loss_D)
        opt_d.step()

        # --- 2. Train G ---
        fake_tgt = self(src)
        fake_pair = torch.cat([src, fake_tgt], dim=1)
        pred_fake = self.discriminator(fake_pair)
        loss_G_GAN = self.loss_fn_gan(pred_fake, torch.ones_like(pred_fake))
        loss_G_L1 = self.loss_fn_l1(fake_tgt, tgt) * self.lambda_L1
        loss_G = loss_G_GAN + loss_G_L1

        opt_g.zero_grad()
        self.manual_backward(loss_G)
        opt_g.step()

        self.log("train/loss_D", loss_D, prog_bar=True)
        self.log("train/loss_G", loss_G, prog_bar=True)
        self.log("train/loss_G_GAN", loss_G_GAN, prog_bar=False)
        self.log("train/loss_G_L1", loss_G_L1, prog_bar=False)

        return {"loss": loss_G + loss_D}

    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        pred = self(src)
        loss = self.loss_fn_l1(pred, tgt)
        self.log("val/loss_L1", loss, prog_bar=True)
        return loss
