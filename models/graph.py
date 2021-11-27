import torch.nn as nn
import torch_geometric.nn as geom_nn

class Graphnet(nn.Module):

    def __init__(self, feed, hidden, out, num_layers=2, layer=geom_nn.GCNConv, drop=0.1, **kwargs):
        """
        Inputs:
            feed - Dimension of input features
            hidden - Dimension of hidden features
            out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for Graph Attention Layer)
        """
        super().__init__()
        gnn_layer = layer

        layers = []
        in_channels, out_channels = feed, hidden
        for lno in range(num_layers-1):
            layers += [
                gnn_layer(in_channels=in_channels,
                          out_channels=out_channels,
                          **kwargs),
                nn.ReLU(inplace=True),
                nn.Dropout(drop)
            ]
            in_channels = hidden
        layers += [gnn_layer(in_channels=in_channels,
                             out_channels=out,
                             **kwargs)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge):
        """
        Inputs:
            x - Input features per node
            edge - List of vertex index pairs representing the edges in the graph
        """
        for l in self.layers:
            if isinstance(l, geom_nn.MessagePassing):
                x = l(x, edge)
            else:
                x = l(x)
        return x