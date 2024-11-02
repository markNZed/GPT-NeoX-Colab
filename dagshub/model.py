# model.py

import pytorch_lightning as pl
from torch import nn, optim

class LitAutoEncoder(pl.LightningModule):
    def __init__(self, hidden_dim, learning_rate):
        """
        Initialize the autoencoder model.

        Args:
            hidden_dim (int): The dimensionality of the hidden layer.
            learning_rate (float): The learning rate for the optimizer.
        """
        super().__init__()
        self.save_hyperparameters()  # Save hyperparameters for checkpointing and logging

        # Encoder: Fully connected layer
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_dim),
            nn.ReLU(),
        )

        # Decoder: Fully connected layer
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 28 * 28),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28)),
        )

    def forward(self, x):
        """
        Forward pass of the autoencoder.

        Args:
            x (Tensor): Input image tensor.
        
        Returns:
            Tensor: Reconstructed image tensor.
        """
        return self.decoder(self.encoder(x))

    def training_step(self, batch, batch_idx):
        """
        Training step for the model.

        Args:
            batch (Tensor): A batch of data.
            batch_idx (int): Batch index.

        Returns:
            Tensor: The calculated loss for the batch.
        """
        x, _ = batch
        x_hat = self(x)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model.

        Args:
            batch (Tensor): A batch of data.
            batch_idx (int): Batch index.

        Returns:
            Tensor: The calculated loss for the batch.
        """
        x, _ = batch
        x_hat = self(x)
        val_loss = nn.functional.mse_loss(x_hat, x)
        self.log("val_loss", val_loss)
        return val_loss

    def configure_optimizers(self):
        """
        Configure optimizers for training.

        Returns:
            Optimizer: The optimizer for training.
        """
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
