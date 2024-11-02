"""
This script demonstrates training an Autoencoder model using PyTorch Lightning, MLFlow, and Hypster for hyperparameter tuning.

Key components:
- **Hypster**: A hyperparameter optimization library that allows dynamic configuration of hyperparameters.
- **MLFlow**: An open-source platform for managing the ML lifecycle, including experimentation, reproducibility, and deployment.
- **DagsHub**: A platform for data versioning and collaboration, which provides data sources for experiments.

The script performs the following steps:
1. Defines an experiment configuration using Hypster.
2. Sets up data loading from DagsHub, applying necessary transformations.
3. Defines an Autoencoder model using PyTorch Lightning.
4. Runs training experiments with different configurations, logging results to MLFlow.

The code is intended for demonstration purposes and showcases how to integrate these tools in a machine learning workflow.
"""

# Import standard libraries
import os
from pathlib import Path

# Import third-party libraries
import mlflow
import pytorch_lightning as pl
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Import Hypster for dynamic hyperparameter configuration
from hypster import HP, config

# Import DagsHub data engine for data sources
from dagshub.data_engine import datasources
import dagshub

# Initialize DagsHub and MLFlow tracking
dagshub.init(repo_name="GPT-NeoX-Colab", repo_owner="MarkNZed")

# Define the transformation pipeline for input images
data_transforms = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet standards
                         std=[0.229, 0.224, 0.225])
])

# Custom Dataset class to handle DagsHub data sources and apply transformations
class CustomDataset(Dataset):
    """
    Custom Dataset class to handle DagsHub data sources and apply transformations.
    """
    def __init__(self, dagshub_datasource, transform=None, download_dir='./downloaded_data'):
        self.datasource = dagshub_datasource
        self.transform = transform
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)  # Ensure download directory exists

    def __len__(self):
        return len(self.datasource)

    def __getitem__(self, idx):
        data_point = self.datasource[idx]
        try:
            # Download the data point's file to the specified local directory
            local_path = data_point.download_file(
                target=str(self.download_dir),
                keep_source_prefix=True,
                redownload=False
            )
            # Open the image using PIL and convert to RGB
            image = Image.open(local_path).convert('RGB')
        except AttributeError as ae:
            raise RuntimeError(f"'Datapoint' object at index {idx} is missing expected attributes: {ae}")
        except Exception as e:
            raise RuntimeError(f"Failed to load image at index {idx}: {e}")

        if self.transform:
            image = self.transform(image)
        return image  # Return only the image tensor

# Define the dynamic experiment configuration using Hypster
@config
def experiment_config(hp: HP):
    # Model hyperparameters
    model = {
        'encoding_channels': hp.select([32, 64], default=32),
        'latent_dim': hp.select([2, 3], default=2)
    }
    # Training hyperparameters
    training = {
        'learning_rate': hp.select([1e-4, 5e-4, 1e-3], default=1e-4),
        'batch_size': hp.select([32, 64], default=32),
        'epochs': 2,
        'limit_train_batches': 50
    }
    # Data configuration
    data = {
        'dataset_name': 'COCO_1K',
        'dataset_owner': 'Dean'
    }

    # Assign configurations to Hypster's HP object
    hp.model = model
    hp.training = training
    hp.data = data

# Define the PyTorch Lightning Module for the Autoencoder model
class LitAutoEncoder(pl.LightningModule):
    """
    PyTorch Lightning Module for the Autoencoder model.
    """
    def __init__(self, model_config, data_config):
        super().__init__()
        self.save_hyperparameters()  # Save hyperparameters for checkpointing and logging

        # Encoder: Convolutional layers to encode the image
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),  # Downsample by a factor of 2
            nn.ReLU(),
            nn.Conv2d(16, model_config['encoding_channels'], kernel_size=4, stride=2, padding=1),  # Downsample again
            nn.ReLU(),
            nn.Conv2d(model_config['encoding_channels'], model_config['latent_dim'], kernel_size=4, stride=2, padding=1),  # Downsample again
            nn.ReLU()
        )

        # Decoder: Transposed convolutional layers to reconstruct the image
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(model_config['latent_dim'], model_config['encoding_channels'], kernel_size=4, stride=2, padding=1),  # Upsample by a factor of 2
            nn.ReLU(),
            nn.ConvTranspose2d(model_config['encoding_channels'], 16, kernel_size=4, stride=2, padding=1),  # Upsample again
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),  # Upsample again to original size
            nn.Sigmoid()  # Output pixel values between 0 and 1
        )

        # Placeholder for learning rate, to be set externally
        self.learning_rate = None

    def training_step(self, batch, batch_idx):
        x = batch  # DataLoader returns only the image tensor
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        if self.learning_rate is None:
            raise ValueError("Learning rate not set. Please set `self.learning_rate` before training.")
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

# Function to run the experiment with a given configuration
def run_experiment(config):
    """
    Runs the training experiment with the given configuration.

    Args:
        config (dict): The configuration dictionary containing model, training, and data settings.
    """
    # Extract configurations
    model_config = config['model']
    training_config = config['training']
    data_config = config['data']

    # Load the dataset from DagsHub and set up DataLoader
    datasource_name = f"{data_config['dataset_owner']}/{data_config['dataset_name']}"
    ds = datasources.get(datasource_name, data_config['dataset_name'])
    custom_dataset = CustomDataset(
        dagshub_datasource=ds.head(),
        transform=data_transforms,
        download_dir='./downloaded_data'  # Specify a directory for downloaded data
    )

    dataset_loader = DataLoader(
        custom_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,                # Shuffle data for training
        num_workers=os.cpu_count(),  # Use all available CPU cores
        pin_memory=True              # Improve GPU transfer speed
    )

    # Initialize the model
    autoencoder = LitAutoEncoder(model_config, data_config)
    autoencoder.learning_rate = training_config['learning_rate']

    # Set up PyTorch Lightning Trainer
    trainer = pl.Trainer(
        limit_train_batches=training_config['limit_train_batches'],
        max_epochs=training_config['epochs'],
        logger=pl.loggers.MLFlowLogger(
            experiment_name="AutoEncoder_Experiments",
            tracking_uri=mlflow.get_tracking_uri()
        ),
        log_every_n_steps=1,
        enable_checkpointing=False  # Disable checkpointing for simplicity
    )

    # Train the model
    trainer.fit(model=autoencoder, train_dataloaders=dataset_loader)

    # Manually log final metrics if needed
    final_train_loss = trainer.callback_metrics.get("train_loss_epoch", None)
    if final_train_loss is not None:
        mlflow.log_metric("final_train_loss", final_train_loss.item())

# Function to run an experiment with specified configuration overrides
def run_experiment_with_config(overrides):
    """
    Runs an experiment with specified configuration overrides.

    Args:
        overrides (dict): A dictionary specifying overrides for the default configuration.
    """
    # Create a specific configuration with overrides
    config = experiment_config(overrides=overrides)
    print("Running Experiment with config:")
    print(config)

    # Start a nested MLflow run for this experiment configuration
    with mlflow.start_run(nested=True):
        # Log parameters within the nested run
        mlflow.log_params({
            "learning_rate": config['training']['learning_rate'],
            "batch_size": config['training']['batch_size'],
            "encoding_channels": config['model']['encoding_channels'],
            "latent_dim": config['model']['latent_dim'],
            "epochs": config['training']['epochs'],
            "limit_train_batches": config['training']['limit_train_batches'],
            "dataset_name": config['data']['dataset_name']
        })

        # Run the experiment within the active MLflow run
        run_experiment(config)

# Function to run multiple experiments with different configurations
def run_all_experiments():
    """
    Runs multiple experiments with different configurations.
    """
    # First configuration with a specific set of hyperparameters
    config_1_overrides = {
        'model.encoding_channels': 32,
        'model.latent_dim': 2,
        'training.learning_rate': 1e-4,
        'training.batch_size': 32
    }

    # Second configuration with a different set of hyperparameters
    config_2_overrides = {
        'model.encoding_channels': 64,
        'model.latent_dim': 3,
        'training.learning_rate': 5e-4,
        'training.batch_size': 64
    }

    # Run the experiments with different configurations
    print("Running first experiment configuration...")
    run_experiment_with_config(config_1_overrides)

    print("\nRunning second experiment configuration...")
    run_experiment_with_config(config_2_overrides)

# Main execution: run all experiments
if __name__ == "__main__":
    run_all_experiments()
