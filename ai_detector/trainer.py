import fire
import lightning as L
from dataset import M4GTDataModule
from hydra import compose, initialize
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from model import LightningClassifier
from utils import get_git_commit


def build_overrides(**kwargs):
    overrides = []
    for key, value in kwargs.items():
        if value is None:
            continue
        overrides.append(f'{key}="{value}"')
    return overrides


def train() -> None:
    with initialize(config_path="../config", version_base=None):
        cfg = compose(
            config_name="config",
        )
    L.seed_everything(cfg.random_state)
    print("started running")
    mlf_logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name,
        tracking_uri=cfg.mlflow.tracking_url,
        tags={"git_commit_id": get_git_commit()},
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="data/checkpoints",
        filename="best-model-loss-{valid_loss:.2f}-epoch-{epoch:02d}",
        save_top_k=1,
    )

    data_module = M4GTDataModule(**cfg.data)

    model_module = LightningClassifier(
        model_config=cfg.model, training_config=cfg.trainer.optimization
    )

    trainer = L.Trainer(
        logger=mlf_logger,
        callbacks=[checkpoint_callback],
        max_epochs=cfg.trainer.max_epochs,
    )
    trainer.fit(model=model_module, datamodule=data_module)
    trainer.fit(model=model_module, datamodule=data_module)
    print("ended running")


def infer(checkpoint_path: str | None = None, output_file: str | None = None):
    overrides = build_overrides(
        checkpoint_path=checkpoint_path,
        output_file=output_file,
    )
    with initialize(config_path="../config", version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    model = LightningClassifier.load_from_checkpoint(cfg.checkpoint_path)
    model.eval()

    data_module = M4GTDataModule(**cfg.data)

    data_module.setup(stage="predict")

    trainer = L.Trainer()
    class_mapping = {0: "human", 1: "machine"}
    preds = trainer.predict(model=model, datamodule=data_module)

    with open("list_output.txt", "w") as file:
        for batch in preds:
            for pred in batch:
                file.write(class_mapping[pred] + "\n")
    print("wrote result to file")


if __name__ == "__main__":
    fire.Fire({"train": train, "infer": infer})
