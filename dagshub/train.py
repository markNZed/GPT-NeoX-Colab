# train.py

import os
import mlflow
import dagshub
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from omegaconf import DictConfig, OmegaConf
import hydra
from model import LitAutoEncoder

# Initialize DagsHub logging
dagshub.init(repo_owner="marknzed", repo_name="GPT-NeoX-Colab", mlflow=True)

def run_experiment(cfg):
    # Print the configuration for debugging
    print(OmegaConf.to_yaml(cfg))

    # Set up data
    transform = transforms.ToTensor()
    train_dataset = MNIST(root=cfg.data.root, train=True, download=True, transform=transform)
    val_dataset = MNIST(root=cfg.data.root, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size)

    # Initialize model with parameters from config
    model = LitAutoEncoder(hidden_dim=cfg.model.hidden_dim, learning_rate=cfg.training.learning_rate)

    # Set up MLFlow logger for DagsHub
    mlflow_logger = MLFlowLogger(experiment_name="AutoEncoder_Experiments")

    # Set up PyTorch Lightning Trainer with checkpointing
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        logger=mlflow_logger,
        callbacks=[pl.callbacks.ModelCheckpoint(
            dirpath="checkpoints/",
            filename="best-checkpoint",
            save_top_k=1,
            monitor="val_loss",
            mode="min"
        )]
    )

    # Train the model
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Log initial parameters
    mlflow.log_params({
        "hidden_dim": cfg.model.hidden_dim,
        "learning_rate": cfg.training.learning_rate,
        "batch_size": cfg.training.batch_size,
        "epochs": cfg.training.epochs
    })
    
    run_experiment(cfg)

if __name__ == "__main__":
    main()
