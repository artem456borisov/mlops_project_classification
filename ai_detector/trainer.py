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
            overrides=[f"random_state={42}"],
        )
    L.seed_everything(cfg.random_state)
    print("started running")
    mlf_logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name,
        tracking_uri=cfg.mlflow.tracking_url,
        tags={"git_commit_id": get_git_commit()},
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-model-loss-{val_loss.2f}-epoch-{epoch}",
        save_top_k=1,
    )

    data_module = M4GTDataModule(
        train_data_dir=cfg.data.train_data_dir,
        val_data_dir=cfg.data.val_data_dir,
        test_data_dir=cfg.data.test_data_dir,
        predict_data_dir=cfg.data.predict_data_dir,
    )

    model_module = LightningClassifier(model_config=cfg.model)

    trainer = L.Trainer(
        logger=mlf_logger,
        callbacks=[checkpoint_callback],
        max_epochs=10,
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

    # Set model to evaluation mode
    model.eval()

    # Prepare data
    data_module = M4GTDataModule(
        train_data_dir=cfg.data.train_data_dir,
        val_data_dir=cfg.data.val_data_dir,
        test_data_dir=cfg.data.test_data_dir,
        predict_data_dir=cfg.data.predict_data_dir,
    )

    # Otherwise use the data module's predict dataloader
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
