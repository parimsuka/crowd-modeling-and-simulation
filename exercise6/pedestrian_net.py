import torch
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl


class PedestrianNet(pl.LightningModule):
#     def __init__(self,
#                  k: int = 10,
#                  hidden_size: int = 3,
#                  learning_rate: float = 1e-3,
#                  optimizer=torch.optim.Adam):
#         super().__init__()
#         self.k = k
#         self.hidden_size = hidden_size
#         self.learning_rate = learning_rate
#         self.optimizer = optimizer

#         self.model = nn.Sequential(
#             nn.Linear(2*k+1, hidden_size, dtype=torch.float64),
#             nn.ReLU(),
#             nn.Linear(hidden_size, 1, dtype=torch.float64),
# #             nn.ReLU()  # We could use ReLU for output, as speed should be in [0, \infty), but let's try without first
#         )
        
    def __init__(self,
             k: int = 10,
             hidden_sizes: list = [3],
             learning_rate: float = 1e-3,
             optimizer=torch.optim.Adam):
        super().__init__()
        self.k = k
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.optimizer = optimizer

        layers = [
            nn.Linear(2 * k + 1, hidden_sizes[0], dtype=torch.float64),
            nn.ReLU()
        ]

        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i], dtype=torch.float64))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_sizes[-1], 1, dtype=torch.float64))
        # layers.append(nn.ReLU())  # We could use ReLU for output, as speed should be in [0, \infty), but let's try without first

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        y_hat = self.model(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        X = batch['distances']
        y = batch['speed'].unsqueeze_(1)
        
        y_hat = self.model(X)
        loss = self._loss_function(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(y))
        return loss

    def validation_step(self, batch, batch_idx):
        X = batch['distances']
        y = batch['speed'].unsqueeze_(1)

        y_hat = self.model(X)
        loss = self._loss_function(y_hat, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(y))

    def test_step(self, batch, batch_idx):
        X = batch['distances']
        y = batch['speed'].unsqueeze_(1)

        y_hat = self.model(X)
        loss = self._loss_function(y_hat, y)
        self.log("TEST_LOSS", loss)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        return optimizer

    def _loss_function(self, y_hat, y):
        return F.mse_loss(y_hat, y)






