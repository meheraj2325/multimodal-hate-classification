import pytorch_lightning as pl
from torch.utils.data import DataLoader

from datasets import FBHateMemeDataset


# Define Lightning DataModule
class FBHateMemeDatamodule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def setup(self, stage=None):
        self.train_dataset = FBHateMemeDataset(self.train_df)
        self.val_dataset = FBHateMemeDataset(self.val_df)
        self.test_dataset = FBHateMemeDataset(self.test_df)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.args.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.args.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.args.batch_size)
