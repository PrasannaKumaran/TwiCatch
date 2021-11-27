import torch
import warnings
import argparse
import torch_geometric
import pytorch_lightning as pl
import torch_geometric.nn as geom_nn
from train import train
pl.seed_everything(42)
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()

CHECKPOINT_PATH = "./data/models"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f'Working on device {device}')

parser.add_argument("-V", "--version", help="show program version", action="store_true")
parser.add_argument("--layer", "-la", nargs='?', help="Graph layer name", default='GCN', )
parser.add_argument("--language", "-ln", nargs='?', help="Streamer language", default='ES')
parser.add_argument("--hidden", "-hd", nargs='?', help="Number of hidden layers", default='8', type=int)
parser.add_argument("--numlayers", "-nl", nargs='?', help="Number of graph layers", default='3', type=int)
parser.add_argument("--drop", "-d", nargs='?', help="Drop rate", default='0.1', type=float)
args = parser.parse_args()
twitch_dataset = torch_geometric.datasets.Twitch(root='./data', name=args.language)

gnn_layer_by_name = {
    "GCN": geom_nn.GCNConv,
    "GAT": geom_nn.GATConv,
    "GraphConv": geom_nn.GraphConv,
    "FastRGCNConv":geom_nn.FastRGCNConv,
    "GatedGraphConv":geom_nn.GatedGraphConv
}

mlp_model, mlp_result = train(model_name="mlp",
                              dataset=twitch_dataset,
                              hidden=args.hidden,
                              num_layers=args.numlayers,
                              drop=args.drop
                            )

gnn_model, gnn_result = train(model_name="GNN",
                              layer = gnn_layer_by_name[args.layer],
                              dataset=twitch_dataset,
                              hidden=args.hidden,
                              num_layers=args.numlayers,
                              drop=args.drop)

print("************ Multi Layer Perceptron Model : ************")
print(f"\t\tTrain accuracy: {(100.0*mlp_result['train']):4.2f}%")
print(f"\t\tVal accuracy:   {(100.0*mlp_result['val']):4.2f}%")
print(f"\t\tTest accuracy:  {(100.0*mlp_result['test']):4.2f}%")

print(f'************ Graph model ({args.layer}) : ************')
print(f"\t\tTrain accuracy: {(100.0*gnn_result['train']):4.2f}%")
print(f"\t\tVal accuracy:   {(100.0*gnn_result['val']):4.2f}%")
print(f"\t\tTest accuracy:  {(100.0*gnn_result['test']):4.2f}%")

