# lightning_train.py

# Import necessary libraries
import os
from torch import optim, nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl  # Correct import for PyTorch Lightning
import mlflow
from hypster import HP, config
from dagshub.data_engine import datasources
import dagshub
from PIL import Image  # Import PIL Image
from pathlib import Path  # For handling file paths

# Initialize DagsHub and MLFlow tracking
dagshub.init(repo_name="GPT-NeoX-Colab", repo_owner="MarkNZed")

# Define the transformation pipeline
data_transforms = transforms.Compose([
    transforms.Resize((480, 640)), 
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize based on ImageNet standards
                         std=[0.229, 0.224, 0.225])
])

# Custom Dataset to apply transforms
class CustomDataset(Dataset):
    def __init__(self, dagshub_datasource, transform=None, download_dir='./downloaded_data'):
        self.datasource = dagshub_datasource
        self.transform = transform
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

    def __len__(self):
        return len(self.datasource)

    def __getitem__(self, idx):
        data_point = self.datasource[idx]
        try:
            # Download the datapoint's file to the specified local directory
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


# Define the dynamic configuration with Hypster
@config
def experiment_config(hp: HP):
    model = {
        'encoding_channels': hp.select([32, 64], default=32),  # Dynamic choice for encoding channels
        'latent_dim': hp.select([2, 3], default=2)             # Dynamic choice for latent dimension
    }
    training = {
        'learning_rate': hp.select([1e-4, 5e-4, 1e-3], default=1e-4),  # Dynamic choice for learning rate
        'batch_size': hp.select([32, 64], default=32),                 # Dynamic choice for batch size
        'epochs': 2,
        'limit_train_batches': 50
    }
    data = {
        'dataset_name': 'COCO_1K',
        'dataset_owner': 'Dean'
    }

    # Assign configurations to Hypster's HP object
    hp.model = model
    hp.training = training
    hp.data = data

# Define the LightningModule model using convolutional layers
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, model_config, data_config):
        super().__init__()
        self.save_hyperparameters()  # Saves hyperparameters for checkpointing

        # Encoder: Convolutional layers
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),      # Halve the spatial dimensions
            nn.ReLU(),
            nn.Conv2d(16, model_config['encoding_channels'], kernel_size=4, stride=2, padding=1),  # Halve again
            nn.ReLU(),
            nn.Conv2d(model_config['encoding_channels'], model_config['latent_dim'], kernel_size=4, stride=2, padding=1),  # Halve again
            nn.ReLU()
        )

        # Decoder: Transposed convolutional layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(model_config['latent_dim'], model_config['encoding_channels'], kernel_size=4, stride=2, padding=1),  # Double spatial dimensions
            nn.ReLU(),
            nn.ConvTranspose2d(model_config['encoding_channels'], 16, kernel_size=4, stride=2, padding=1),  # Double again
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),  # Double again
            nn.Sigmoid()  # Output pixel values between 0 and 1
        )
        self.learning_rate = None  # Will be set externally

    def training_step(self, batch, batch_idx):
        x = batch  # DataLoader now returns only the image tensor
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

# Define the experiment function
import mlflow

# ... [Previous code remains unchanged]

def run_experiment(config):
    model_config = config['model']
    training_config = config['training']
    data_config = config['data']

    # Log hyperparameters manually
    mlflow.log_params({
        "learning_rate": training_config['learning_rate'],
        "batch_size": training_config['batch_size'],
        "encoding_channels": model_config['encoding_channels'],
        "latent_dim": model_config['latent_dim'],
        "epochs": training_config['epochs'],
        "limit_train_batches": training_config['limit_train_batches'],
        "dataset_name": data_config['dataset_name']
    })

    # Load the dataset from DagsHub and set up DataLoader
    ds = datasources.get(f"{data_config['dataset_owner']}/{data_config['dataset_name']}", data_config['dataset_name'])
    custom_dataset = CustomDataset(
        dagshub_datasource=ds.head(),
        transform=data_transforms,
        download_dir='./downloaded_data'  # Specify a directory with sufficient space
    )

    dataset_loader = DataLoader(
        custom_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,                # Shuffle for training
        num_workers=os.cpu_count(),  # Dynamically set based on CPU cores
        pin_memory=True              # Improves GPU transfer speed
    )

    # Initialize and train the model without MLflow’s autologging
    autoencoder = LitAutoEncoder(model_config, data_config)
    autoencoder.learning_rate = training_config['learning_rate']

    trainer = pl.Trainer(
        limit_train_batches=training_config['limit_train_batches'],
        max_epochs=training_config['epochs'],
        logger=pl.loggers.MLFlowLogger(
            experiment_name="AutoEncoder_Experiments",
            tracking_uri=mlflow.get_tracking_uri()
        ),
        log_every_n_steps=1,
        enable_checkpointing=False  # Ensure no checkpointing
    )

    # Train the model
    trainer.fit(model=autoencoder, train_dataloaders=dataset_loader)

    # Manually log final metrics if needed
    final_train_loss = trainer.callback_metrics.get("train_loss_epoch", None)
    if final_train_loss is not None:
        mlflow.log_metric("final_train_loss", final_train_loss.item())

def run_experiment_with_config(overrides):
    # Create a specific configuration with overrides
    config = experiment_config(overrides=overrides)
    print("Running Experiment with config:")
    print(config)
    
    # Start a nested MLflow run for this experiment configuration
    with mlflow.start_run(nested=True):
        # Log parameters within the nested run
        try:
            mlflow.log_params({
                "learning_rate": config['training']['learning_rate'],
                "batch_size": config['training']['batch_size'],
                "encoding_channels": config['model']['encoding_channels'],
                "latent_dim": config['model']['latent_dim'],
                "epochs": config['training']['epochs'],
                "limit_train_batches": config['training']['limit_train_batches'],
                "dataset_name": config['data']['dataset_name']
            })
        except mlflow.exceptions.MlflowException as e:
            print(f"Failed to log parameter: {e}")
        
        # Run the experiment within the active MLflow run
        run_experiment(config)

def run_all_experiments():
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

# Run experiments with different configurations
if __name__ == "__main__":
    run_all_experiments()
