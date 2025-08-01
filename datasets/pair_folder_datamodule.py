# datasets/pair_folder_datamodule.py
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pathlib, random

class PairFolder(Dataset):
    def __init__(self, root, img_size=256, split="train"):
        self.root = pathlib.Path(root)
        self.un_dir = self.root / "unstained"
        self.st_dir = self.root / "stained"
        self.files  = sorted(p.name for p in self.un_dir.glob("*.png"))
        random.seed(0)
        random.shuffle(self.files)
        split_idx = int(0.8*len(self.files))
        self.files = self.files[:split_idx] if split=="train" else self.files[split_idx:]
        self.t = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        name = self.files[idx]
        src = self.t(Image.open(self.un_dir/name).convert("RGB"))
        tgt = self.t(Image.open(self.st_dir/name).convert("RGB"))
        return src, tgt

class PairFolderDataModule(pl.LightningDataModule):
    def __init__(self, data_root, img_size=256, batch_size=4, num_workers=4):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        self.train_set = PairFolder(self.hparams.data_root, self.hparams.img_size, "train")
        self.val_set   = PairFolder(self.hparams.data_root, self.hparams.img_size, "val")

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.hparams.batch_size,
                          shuffle=True, num_workers=self.hparams.num_workers,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          persistent_workers=True)
