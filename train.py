from dataset import *
import logging
import gin
import gin.torch
import gin.torch.external_configurables
from torch.utils.tensorboard import SummaryWriter
import torch_geometric
import torch


logger = logging.getLogger("train")
DataLoader = gin.external_configurable(torch_geometric.loader.DataLoader,\
    module="torch_geometric.loader")
BCEWithLogitsLoss = gin.external_configurable(torch.nn.BCEWithLogitsLoss,\
    module="torch.nn")

import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from dataclasses import dataclass


def dataNNModule(_dcls):
    class cls(torch.nn.Module, _dcls):
        def __init__(self, *args, **kwargs):
            torch.nn.Module.__init__(self)
            _dcls.__init__(self, *args, **kwargs)
    return cls

# @gin.configurable
class GCN(torch.nn.Module):
    def __init__(self,
        hidden_channels: int,
        num_features: int,
        num_outputs: int,
        dropout_p: float,
        add_self_loops: bool = False,
        ):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels, add_self_loops=add_self_loops)
        self.conv2 = GCNConv(hidden_channels, num_outputs, add_self_loops=add_self_loops)
        self.dropout_p = dropout_p


    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def load_dataset():
    dataset = SceneGraphChangeDataset();
    return dataset;


def save_config_files(log_dir, config_files):
    import shutil
    for f in config_files:
        shutil.copy(f, log_dir)


@gin.configurable
def training_main(
        config_files: list[str],
        num_epochs: int,
        experiment_name: str,
        # trainable_cls: torch.nn.Module,
        optimizer_cls,
        loss_cls,
        loss_params: dict,
    ):

    log_dir=f'~/3dsg_runs/{experiment_name}'
    writer = SummaryWriter(log_dir=log_dir)  # creates dir?
    writer.add_text(f"gin_config/{experiment_name}", gin.operative_config_str())
    writer.flush()
    save_config_files(log_dir, config_files)
    log_file = logging.FileHandler(f"{log_dir}/stdout.log")
    log_file.setLevel("DEBUG")
    logger.addHandler(log_file)

    dataset = load_dataset()
    one_graph = dataset[0]
    num_node_attrs = one_graph.x.shape[-1]
    num_edge_attrs = one_graph.edge_attr.shape[-1]
    num_outputs = one_graph.y.shape[-1]
    logger.warning(experiment_name)
    logger.info(f"dataset configured to have {num_node_attrs=}, {num_edge_attrs=}, {num_outputs=}")
    loader = DataLoader(dataset)  # additional params in config file
    trainable = GCN(hidden_channels=32, num_features=num_node_attrs, num_outputs=num_outputs, dropout_p=0.5, add_self_loops=False)
    optimizer = optimizer_cls(trainable.parameters())
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = loss_cls(**loss_params)
    for epoch in range(1, num_epochs+1):
        logger.warning(f"{epoch=}")
        trainable.train()
        for batch_num, batch in enumerate(loader):
            logger.info(f"{epoch=}{batch_num=}")
            logger.debug(f"graph {batch.input_graph}")
            optimizer.zero_grad()
            out = trainable(batch.x, batch.edge_index)
            loss = criterion(out, batch.y)
            writer.add_scalar(f"Loss/train", loss, epoch)
            writer.flush()
            loss.backward()
            optimizer.step()
    # writer.add_graph(trainable)
    writer.close()
    return trainable


if __name__ == "__main__":
    import sys
    logging.basicConfig(level='INFO')
    gin_config_files = sys.argv[1:]
    logger.debug(f"using {gin_config_files=}")
    gin.parse_config_files_and_bindings(gin_config_files, "")
    training_main(config_files=gin_config_files)
