import torch
import torch.nn as nn
from core.registry import register
from core.base_model import BaseVSModel  # Adjust path as needed
import itertools

# ---- Generators and Discriminators ----
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False):
        super().__init__()
        p = 0
        if padding_type == 'reflect':
            self.padding1 = nn.ReflectionPad2d(1)
            self.padding2 = nn.ReflectionPad2d(1)
        elif padding_type == 'replicate':
            self.padding1 = nn.ReplicationPad2d(1)
            self.padding2 = nn.ReplicationPad2d(1)
        elif padding_type == 'zero':
            self.padding1 = nn.Identity()
            self.padding2 = nn.Identity()
            p = 1
        else:
            raise NotImplementedError()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=p)
        self.norm1 = norm_layer(dim)
        self.relu = nn.ReLU(True)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=p)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        y = self.padding1(x)
        y = self.conv1(y)
        y = self.norm1(y)
        y = self.relu(y)
        if self.use_dropout:
            y = self.dropout(y)
        y = self.padding2(y)
        y = self.conv2(y)
        y = self.norm2(y)
        return x + y

class ResnetGenerator(nn.Module):
    """ResNet-based generator (default: 9 blocks for 256x256)"""
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9, norm_layer=nn.BatchNorm2d):
        assert(n_blocks >= 0)
        super().__init__()
        use_bias = True
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        ]
        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)
            ]
        # Resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer, use_dropout=False)]
        # Upsampling
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super().__init__()
        use_bias = True
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)

# --- Actual CycleGAN Model ---

@register
class CycleGAN(BaseVSModel):
    """CycleGAN model for unpaired image-to-image virtual staining.
    Manual optimization, two optimizers (G and D), modular style."""
    def __init__(self, hparams):
        super().__init__(hparams)
        hp = hparams if isinstance(hparams, dict) else vars(hparams)
        self.input_nc = hp.get("input_nc", 3)
        self.output_nc = hp.get("output_nc", 3)
        self.ngf = hp.get("ngf", 64)
        self.ndf = hp.get("ndf", 64)
        self.n_blocks = hp.get("n_blocks", 9)
        self.n_layers_D = hp.get("n_layers_D", 3)
        self.lambda_A = hp.get("lambda_A", 10.0)
        self.lambda_B = hp.get("lambda_B", 10.0)
        self.lambda_identity = hp.get("lambda_identity", 0.5)
        self.lr = hp.get("lr", 2e-4)
        self.beta1 = hp.get("beta1", 0.5)
        self.automatic_optimization = False  # manual opt for GAN

        # Two generators: A→B and B→A
        self.netG_A = ResnetGenerator(self.input_nc, self.output_nc, ngf=self.ngf, n_blocks=self.n_blocks)
        self.netG_B = ResnetGenerator(self.output_nc, self.input_nc, ngf=self.ngf, n_blocks=self.n_blocks)
        # Two discriminators: D_A (for B), D_B (for A)
        self.netD_A = PatchGANDiscriminator(self.output_nc, ndf=self.ndf, n_layers=self.n_layers_D)
        self.netD_B = PatchGANDiscriminator(self.input_nc, ndf=self.ndf, n_layers=self.n_layers_D)

        # Losses
        self.loss_fn_gan = nn.MSELoss()
        self.loss_fn_cycle = nn.L1Loss()
        self.loss_fn_idt = nn.L1Loss()

    def forward(self, x, direction='AtoB'):
        # By convention, x is from domain A if direction='AtoB', B if 'BtoA'
        if direction == 'AtoB':
            return self.netG_A(x)
        else:
            return self.netG_B(x)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
            lr=self.lr, betas=(self.beta1, 0.999))
        opt_d = torch.optim.Adam(
            itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
            lr=self.lr, betas=(self.beta1, 0.999))
        return [opt_g, opt_d]

    def training_step(self, batch, batch_idx):
        # NOTE: expects unpaired batches: src is domain A, tgt is domain B
        src, tgt = batch  # src: A, tgt: B
        opt_g, opt_d = self.optimizers()

        # -- Generators forward --
        fake_B = self.netG_A(src)
        rec_A = self.netG_B(fake_B)
        fake_A = self.netG_B(tgt)
        rec_B = self.netG_A(fake_A)

        # -- Identity loss --
        idt_A = self.netG_A(tgt)
        idt_B = self.netG_B(src)
        loss_idt_A = self.loss_fn_idt(idt_A, tgt) * self.lambda_B * self.lambda_identity
        loss_idt_B = self.loss_fn_idt(idt_B, src) * self.lambda_A * self.lambda_identity

        # -- GAN loss --
        pred_fake_B = self.netD_A(fake_B)
        target_real_B = torch.ones_like(pred_fake_B)
        loss_G_A = self.loss_fn_gan(pred_fake_B, target_real_B)

        pred_fake_A = self.netD_B(fake_A)
        target_real_A = torch.ones_like(pred_fake_A)
        loss_G_B = self.loss_fn_gan(pred_fake_A, target_real_A)

        # -- Cycle loss --
        loss_cycle_A = self.loss_fn_cycle(rec_A, src) * self.lambda_A
        loss_cycle_B = self.loss_fn_cycle(rec_B, tgt) * self.lambda_B

        # -- Full generator loss --
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B

        opt_g.zero_grad()
        self.manual_backward(loss_G)
        opt_g.step()

        # -- Discriminator A --
        pred_real_B = self.netD_A(tgt)
        loss_D_A_real = self.loss_fn_gan(pred_real_B, torch.ones_like(pred_real_B))
        pred_fake_B = self.netD_A(fake_B.detach())
        loss_D_A_fake = self.loss_fn_gan(pred_fake_B, torch.zeros_like(pred_fake_B))
        loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5

        # -- Discriminator B --
        pred_real_A = self.netD_B(src)
        loss_D_B_real = self.loss_fn_gan(pred_real_A, torch.ones_like(pred_real_A))
        pred_fake_A = self.netD_B(fake_A.detach())
        loss_D_B_fake = self.loss_fn_gan(pred_fake_A, torch.zeros_like(pred_fake_A))
        loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5

        loss_D = loss_D_A + loss_D_B

        opt_d.zero_grad()
        self.manual_backward(loss_D)
        opt_d.step()

        # Logging
        self.log("train/loss_G", loss_G, prog_bar=True)
        self.log("train/loss_D", loss_D, prog_bar=True)
        self.log("train/loss_cycle_A", loss_cycle_A)
        self.log("train/loss_cycle_B", loss_cycle_B)
        self.log("train/loss_idt_A", loss_idt_A)
        self.log("train/loss_idt_B", loss_idt_B)
        self.log("train/loss_G_A", loss_G_A)
        self.log("train/loss_G_B", loss_G_B)
        self.log("train/loss_D_A", loss_D_A)
        self.log("train/loss_D_B", loss_D_B)

        return {"loss": loss_G + loss_D}

    def validation_step(self, batch, batch_idx):
        # Unpaired validation: show cycle consistency and identity
        src, tgt = batch
        fake_B = self.netG_A(src)
        rec_A = self.netG_B(fake_B)
        loss_cycle_A = self.loss_fn_cycle(rec_A, src)
        fake_A = self.netG_B(tgt)
        rec_B = self.netG_A(fake_A)
        loss_cycle_B = self.loss_fn_cycle(rec_B, tgt)
        self.log("val/loss_cycle_A", loss_cycle_A, prog_bar=True)
        self.log("val/loss_cycle_B", loss_cycle_B, prog_bar=True)
        return (loss_cycle_A + loss_cycle_B) / 2
