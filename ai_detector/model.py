import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import (BinaryF1Score, BinaryPrecision,
                                         BinaryRecall)


class TextsClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with appropriate scaling"""
        for module in [self.fc1, self.fc2]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.norm(x)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return torch.sigmoid(self.fc2(x))


class LightningClassifier(L.LightningModule):
    def __init__(self, model_config: dict, training_config: dict):
        super().__init__()
        self.model = TextsClassifier(**model_config)
        self.criterion = nn.BCELoss()
        self.training_config = training_config

        self.train_precision = BinaryPrecision()
        self.train_recall = BinaryRecall()
        self.train_f1 = BinaryF1Score()

        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()
        self.val_f1 = BinaryF1Score()

    def forward(self, inputs):
        probs = self.model(inputs)
        return probs

    def training_step(self, batch):
        inputs, target = batch
        output = self.model(inputs)
        loss = self.criterion(output.reshape(-1), target)

        preds = (output > 0.5).int().flatten()
        self.train_precision(preds, target.int())
        self.train_recall(preds, target.int())
        self.train_f1(preds, target.int())

        accuracy = (preds == target).float().mean()

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_accuracy", accuracy, prog_bar=True, on_epoch=True)

        self.log("train_precision", self.train_precision, prog_bar=False, on_epoch=True)
        self.log("train_recall", self.train_recall, prog_bar=False, on_epoch=True)
        self.log("train_f1", self.train_f1, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, batch):
        inputs, target = batch
        output = self.model(inputs)
        loss = self.criterion(output.reshape(-1), target)

        preds = (output > 0.5).int().flatten()

        self.val_precision(preds, target.int())
        self.val_recall(preds, target.int())
        self.val_f1(preds, target.int())
        accuracy = (preds == target).float().mean()

        self.log("valid_loss", loss, prog_bar=True, on_epoch=True)
        self.log("valid_accuracy", accuracy, prog_bar=True, on_epoch=True)

        self.log("valid_precision", self.val_precision, on_epoch=True)
        self.log("valid_recall", self.val_recall, on_epoch=True)
        self.log("valid_f1", self.val_f1, prog_bar=True, on_epoch=True)

    def predict_step(self, batch):
        inputs, target = batch
        output = self.model(inputs)
        preds = (output > 0.5).int()
        return preds.flatten().tolist()

    def on_before_optimizer_step(self, optimizer):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

    def configure_optimizers(self):
        if self.training_config.optimizer == "AdamW":
            return torch.optim.AdamW(
                self.model.parameters(), lr=self.training_config.lr
            )
        else:
            return torch.optim.SGD(self.model.parameters(), lr=self.training_config.lr)
