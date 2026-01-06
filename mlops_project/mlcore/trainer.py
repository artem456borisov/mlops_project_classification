import fire
import hydra
import lightning as L
from dataset import M4GTDataModule
from lightning.pytorch.loggers import MLFlowLogger
from model import LightningClassifier
from omegaconf import DictConfig
from utils import get_git_commit

# from lightning.pytorch.callbacks import ModelCheckpoint


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    L.seed_everything(cfg.random_state)
    print("started running")
    mlf_logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name,
        tracking_uri=cfg.mlflow.tracking_url,
        tags={"git_commit_id": get_git_commit()},
    )

    # checkpoint_callback = ModelCheckpoint(
    #     dirpath='checkpoints',
    #     filename='best-model-loss-{val_loss.2f}-epoch-{epoch}',
    #     save_top_k=1
    # )

    data_module = M4GTDataModule(
        train_data_dir=cfg.data.train_data_dir,
        val_data_dir=cfg.data.val_data_dir,
        test_data_dir=cfg.data.test_data_dir,
        predict_data_dir=cfg.data.predict_data_dir,
    )

    model_module = LightningClassifier()

    trainer = L.Trainer(
        logger=mlf_logger,
        # callbacks=[checkpoint_callback],
        max_epochs=10,
    )
    trainer.fit(model=model_module, datamodule=data_module)
    trainer.fit(model=model_module, datamodule=data_module)
    print("ended running")


if __name__ == "__main__":
    fire.Fire(main)
