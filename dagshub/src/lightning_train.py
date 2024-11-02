# Import necessary libraries
import os
from torch import optim, nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl  # Corrected import for PyTorch Lightning
import mlflow
from hypster import HP, config
from dagshub.data_engine import datasources
import dagshub

# Initialize DagsHub and MLFlow tracking
dagshub.init(repo_name="GPT-NeoX-Colab", repo_owner="MarkNZed")
mlflow.autolog()  # Enable MLflow autologging

# Define the transformation pipeline
data_transforms = transforms.Compose([
    transforms.Resize((480, 640)),                                # Resize images to 480x640
    transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),  # Ensure 3 channels
    transforms.ToTensor(),                                        # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],              # Normalize based on ImageNet standards
                         std=[0.229, 0.224, 0.225])
])

# Custom Dataset to apply transforms
class CustomDataset(Dataset):
    def __init__(self, dagshub_datasource, transform=None):
        self.datasource = dagshub_datasource
        self.transform = transform

    def __len__(self):
        return len(self.datasource)

    def __getitem__(self, idx):
        data = self.datasource[idx]
        # Adjust based on how data is returned; assuming (image, label)
        image, label = data if isinstance(data, tuple) else (data, None)
        if self.transform:
            image = self.transform(image)
        return image, label

# Define the dynamic configuration with Hypster
@config
def experiment_config(hp: HP):
    model = {
        'input_dim': 480 * 640 * 3,
        'encoding_dim': hp.select([32, 64], default=32),      # Dynamic choice for encoding dimension
        'latent_dim': hp.select([2, 3], default=2)            # Dynamic choice for latent dimension
    }
    training = {
        'learning_rate': hp.select([1e-4, 5e-4, 1e-3], default=1e-4),  # Use hp.select instead of number_input
        'batch_size': hp.select([32, 64], default=32),               # Dynamic choice for batch size
        'epochs': 1,
        'limit_train_batches': 50
    }
    data = {
        'dataset_name': 'COCO_1K',
        'dataset_owner': 'Dean',
        'image_size': (480, 640)
    }
    
    # Assign configurations to Hypster's HP object
    hp.model = model
    hp.training = training
    hp.data = data

# Define the LightningModule model that can use Hypster config values
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, model_config, data_config):
        super().__init__()
        self.save_hyperparameters()  # Saves hyperparameters for checkpointing
        
        self.encoder = nn.Sequential(
            nn.Linear(model_config['input_dim'], model_config['encoding_dim']), 
            nn.ReLU(), 
            nn.Linear(model_config['encoding_dim'], model_config['latent_dim'])
        )
        self.decoder = nn.Sequential(
            nn.Linear(model_config['latent_dim'], model_config['encoding_dim']), 
            nn.ReLU(), 
            nn.Linear(model_config['encoding_dim'], model_config['input_dim'])
        )
        # Removed transform from here since it's handled by the DataLoader
        self.learning_rate = None  # Will be set externally

    def training_step(self, batch, batch_idx):
        x, _ = batch  # Assuming DataLoader returns (image, label). Ignore label if not needed.
        x = x.view(x.size(0), -1)  # Flatten the images
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
def run_experiment(config):
    # Access configurations from Hypster's config dict
    model_config = config['model']
    training_config = config['training']
    data_config = config['data']

    # Load the dataset from DagsHub
    ds = datasources.get(f"{data_config['dataset_owner']}/{data_config['dataset_name']}")

    # Wrap the datasource with the custom dataset and apply transforms
    custom_dataset = CustomDataset(ds.head(), transform=data_transforms)

    # Setup the DataLoader with the selected batch size
    dataset_loader = DataLoader(
        custom_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,          # Shuffle for training
        num_workers=4,         # Adjust based on your CPU cores
        pin_memory=True        # Improves GPU transfer speed
    )

    # Initialize the model with the model and data configurations
    autoencoder = LitAutoEncoder(model_config, data_config)
    autoencoder.learning_rate = training_config['learning_rate']  # Set the learning rate dynamically

    # Log additional parameters if needed (optional)
    mlflow.log_params({
        "learning_rate": training_config['learning_rate'],
        "batch_size": training_config['batch_size'],
        "encoding_dim": model_config['encoding_dim'],
        "latent_dim": model_config['latent_dim'],
        "epochs": training_config['epochs'],
        "limit_train_batches": training_config['limit_train_batches'],
        "dataset_name": data_config['dataset_name'],
        "image_size": data_config['image_size']
    })

    # Initialize the Trainer with MLFlow Logger (autologging should handle this)
    trainer = pl.Trainer(
        limit_train_batches=training_config['limit_train_batches'],
        max_epochs=training_config['epochs'],
        logger=pl.loggers.MLFlowLogger()  # Ensure MLFlow logger is used
    )
    
    # Train the model
    trainer.fit(model=autoencoder, train_dataloaders=dataset_loader)
    
    # Optionally, log artifacts or additional metrics here

# Experiment runner to create and run experiments using dynamic configurations
def run_all_experiments():
    # Retrieve the full configuration
    config = experiment_config()
    print("Running Experiment with config:")
    print(config)
    run_experiment(config)

# Run experiments with dynamic configurations
if __name__ == "__main__":
    run_all_experiments()
