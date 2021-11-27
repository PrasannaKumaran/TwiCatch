import os
import torch
import pytorch_lightning as pl
import torch_geometric.data as geom_data
from models.node import NodeLevel
from pytorch_lightning.callbacks import  ModelCheckpoint

CHECKPOINT_PATH = "./data/models"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def train(model_name, dataset, **model_kwargs):
    
    node_data_loader = geom_data.DataLoader(dataset, batch_size=32)
    root_dir = os.path.join(CHECKPOINT_PATH, "NodePrediction" + model_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
                         gpus=1 if str(device).startswith("cuda") else 0,
                         max_epochs=200,
                         progress_bar_refresh_rate=0) 

    pretrained = os.path.join(CHECKPOINT_PATH, f"NodePrediction{model_name}.ckpt")
    if os.path.isfile(pretrained):
        print("Pretrained model found!, Initiating")
        model = NodeLevel.load_from_checkpoint(pretrained)
    else:
        model = NodeLevel(
            model_name=model_name,
            feed=dataset.num_node_features,
            out=dataset.num_classes,
            **model_kwargs
        )
        trainer.fit(model, node_data_loader, node_data_loader)
        model = NodeLevel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    test_result = trainer.test(model, test_dataloaders=node_data_loader, verbose=False)
    batch = next(iter(node_data_loader))
    batch = batch.to(model.device)
    _, train_acc = model.forward(batch, mode="train")
    _, val_acc = model.forward(batch, mode="val")
    result = {"train": train_acc,
              "val": val_acc,
              "test": test_result[0]['test_acc']}
    return model, result
