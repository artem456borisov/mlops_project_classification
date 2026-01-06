import torch
from dataset import M4GTDataModule

# Force CPU instead of MPS
torch.set_default_device("cpu")

data_module = M4GTDataModule(
    train_data_dir="data_correct.json",
    val_data_dir="data_correct.json",
    test_data_dir="data_correct.json",
    predict_data_dir="data_correct.json",
)

from model import LigthningClassifier

module = LigthningClassifier()

import lightning as L

trainer = L.Trainer()

trainer.fit(module, datamodule=data_module)
