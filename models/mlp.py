import torch.nn as nn
class MultiLayerPerceptron(nn.Module):

    def __init__(self, feed, hidden, out, num_layers=2, drop=0.1):
        """
        Inputs:
            feed - Dimension of input features
            hidden - Dimension of hidden features
            out - Dimension of the output features.
            num_layers - Number of hidden layers
            drop - Dropout rate 
        """
        super().__init__()
        layers = []
        in_channels, out_channels = feed, hidden
        for lno in range(num_layers-1):
            layers += [
                nn.Linear(in_channels, out_channels),
                nn.Tanh(),
                nn.Dropout(drop)
            ]
            in_channels = hidden
        layers += [nn.Linear(in_channels, out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x, *args, **kwargs):
        """
        Inputs:
            x - Input features per node
        """
        return self.layers(x)