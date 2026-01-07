# import json
import dill
import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset


class M4GTDataset(Dataset):
    def __init__(self, data_path: str):
        try:
            with open(data_path, "rb") as f:
                self.data = dill.load(f)
                self.data = self.data.dropna(subset=["embeddings"])

        except Exception as e:
            print(f"Unable to read pkl: {e}")

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, index):
        return self.data.iloc[index]["embeddings"], torch.tensor(
            self.data.iloc[index]["label"], dtype=torch.float32
        )


class M4GTDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_data_dir: str = None,
        val_data_dir: str = None,
        test_data_dir: str = None,
        predict_data_dir: str = None,
        train_batch_size: int = 64,
        predict_batch_size: int = 128,
    ) -> None:
        super().__init__()
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.test_data_dir = test_data_dir
        self.predict_data_dir = predict_data_dir
        self.train_batch_size = train_batch_size

    def setup(self, stage):
        if stage == "fit":
            self.train_dataset = M4GTDataset(self.train_data_dir)
            self.val_dataset = M4GTDataset(self.val_data_dir)
        elif stage == "validate":
            self.val_dataset = M4GTDataset(self.val_data_dir)
        elif stage == "test":
            self.test_dataset = M4GTDataset(self.test_data_dir)
        elif stage == "predict":
            self.predict_dataset = M4GTDataset(self.predict_data_dir)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.train_batch_size, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.train_batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.train_batch_size)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.train_batch_size)
