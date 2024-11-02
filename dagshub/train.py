import os
import mlflow
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
from model import LitAutoEncoder  # Import the model

# Ensure the MLFlow directory exists
mlruns_path = "./mlruns"
os.makedirs(mlruns_path, exist_ok=True)
mlflow.set_tracking_uri(f"file:{mlruns_path}")

def run_experiment(cfg):
    # Print the configuration for debugging
    print(OmegaConf.to_yaml(cfg))

    # Set up data
    transform = transforms.ToTensor()
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True)

    # Initialize model with parameters from config
    model = LitAutoEncoder(hidden_dim=cfg.model.hidden_dim, learning_rate=cfg.training.learning_rate)

    # Ensure the experiment exists or create it if it doesn't
    experiment_name = "AutoEncoder_Experiments"
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    # Set up MLFlow logger
    mlflow_logger = MLFlowLogger(experiment_name=experiment_name)

    # Set up PyTorch Lightning Trainer
    trainer = pl.Trainer(max_epochs=cfg.training.epochs, logger=mlflow_logger)

    # Start MLFlow run
    with mlflow.start_run():
        # Log hyperparameters to MLFlow
        mlflow.log_params({
            'hidden_dim': cfg.model.hidden_dim,
            'learning_rate': cfg.training.learning_rate,
            'batch_size': cfg.training.batch_size
        })

        # Train the model
        trainer.fit(model, train_loader)

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Run the experiment
    run_experiment(cfg)

if __name__ == "__main__":
    # Run Experiment 1
    main()

    # Run Experiment 2 by overriding the experiment configuration
    from hydra import initialize, compose

    with initialize(config_path="configs"):
        cfg = compose(config_name="config", overrides=["experiment=experiment2"])
        run_experiment(cfg)
