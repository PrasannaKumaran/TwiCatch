import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from .mlp import MultiLayerPerceptron
from .graph import Graphnet
class NodeLevel(pl.LightningModule):

    def __init__(self, model_name, **model_kwargs):
        super().__init__()
        self.save_hyperparameters()

        if model_name == "mlp":
            self.model = MultiLayerPerceptron(**model_kwargs)
        else:
            self.model = Graphnet(**model_kwargs)
        self.loss_module = nn.CrossEntropyLoss()

    def forward(self, data, mode="train"):
        x, edge_index = data.x, data.edge_index
        x = self.model(x, edge_index)
        loss = self.loss_module(x, data.y)
        acc = (x.argmax(dim=-1) == data.y).sum().float() / x.shape[0]
        return loss, acc

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="train")
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="val")
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="test")
        self.log('test_acc', acc)