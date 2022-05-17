#!/usr/bin/env python3
import torch
from torch.utils.tensorboard import SummaryWriter
import torch_geometric
from dataset import *
from dataset.SceneGraphChangeDataset import SceneGraphChangeDataset, transform_locations
from dataset.utils.extract_data import build_scene_graph
from dataset.utils.logging import TrainingLogger
from dataset.DatasetCfg import DatasetCfg
from models.SimpleGCN import SimpleMPGNN
from models.SimpleMLP import SimpleMLP
from models.FocalLoss import FocalLoss
from scripts.eval import calculate_conf_mat
import numpy as np
from tqdm import tqdm
import logging
import gin
import gin.torch.external_configurables
import pickle

global_config_str: str = ""
gin_config_files = []
stdout_str = ""


stdout = logging.getLogger("train_variability_ablation")


def save_config_files(log_dir, config_files):
    import shutil
    for f in config_files:
        shutil.copy(f, log_dir)


DataLoader = gin.external_configurable(torch_geometric.loader.DataLoader,\
    module="torch_geometric.loader")

def calculate_training_loss(data, nnet, l_fn, nnet_type):
    x_i = data.x
    node_mask = data.y[:, 2]
    state_mask = data.state_mask
    if nnet_type == "gnn":
        e_idx_i = data.edge_index.type(torch.LongTensor)
        e_att_i = data.edge_attr
        pred_i = nnet(x_i, e_idx_i, e_att_i)
    else:
        pred_i = nnet(x_i)
    loss_state = l_fn(pred_i[:, 0], data.y[:, 0])
    loss_pos = l_fn(pred_i[:, 1], data.y[:, 1])
    loss_mask = l_fn(pred_i[:, 2], data.y[:, 2])
    loss = (torch.sum(torch.multiply(loss_state, state_mask)) + torch.sum(torch.multiply(loss_pos, node_mask)) + torch.sum(loss_mask)) / (torch.sum(node_mask) + torch.mean(loss_mask) + torch.numel(node_mask))
    return loss, pred_i


@gin.configurable
def train_neuralnet(dset,
                    neuralnet,
                    base_log_dir: str,
                    experiment_name: str,
                    epochs: int,
                    nnet_type: str,
                    bs: int,
                    seed: int,
                    l_fn = FocalLoss(torch.ones(1,3), 0.5),
                    visualize=False):

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    log_dir=f'{base_log_dir}/{experiment_name}/{nnet_type}'
    stdout.warning(experiment_name)
    stdout.warning(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    global global_config_str
    writer.add_text(f"gin_config", global_config_str)
    writer.flush()
    save_config_files(log_dir, gin_config_files)
    log_file = logging.FileHandler(f"{log_dir}/stdout.log")
    log_file.setLevel("DEBUG")
    stdout.addHandler(log_file)

    train_n = int(len(dset) * 0.85)
    train_set, val_set = torch.utils.data.random_split(dset,  [train_n, len(dset)-train_n])
    train_loader = DataLoader(train_set, batch_size=bs)
    val_loader = DataLoader(val_set, batch_size=1)
    optimizer = torch.optim.Adam(neuralnet.parameters())

    logger = TrainingLogger(experiment_name, print_interval=20, train_n=int(train_n/bs))
    for i in range(epochs):
        stdout.warning("Training. Epoch {}".format(i+1))
        neuralnet.train()
        for data in train_loader:
            optimizer.zero_grad()
            loss, _ = calculate_training_loss(data, neuralnet, l_fn, nnet_type)
            # writer.add_scalar("Loss/train", loss, i)
            logger.log_training_iter(loss.item())
            loss.backward()
            optimizer.step()
        logger.log_training_epoch(neuralnet)
        torch.save(neuralnet, f"{log_dir}/model_epoch_{i}.pt")


        stdout.warning("Validation:")
        neuralnet.eval()
        for data in tqdm(val_loader):
            loss, pred_i = calculate_training_loss(data, neuralnet, l_fn, nnet_type)
            val_loss = loss.item()
            state_conf = calculate_conf_mat(pred_i[:, 0], data.y[:, 0])
            pos_conf = calculate_conf_mat(pred_i[:, 1], data.y[:, 1])
            mask_conf = calculate_conf_mat(pred_i[:, 2], data.y[:, 1])
            logger.log_validation_iter(val_loss, state_conf, pos_conf, mask_conf)

            if visualize:
                _, nodes_1, edges_1 = build_scene_graph(train_set.dataset.objects_dict, train_set.dataset.relationships_dict, data[0].input_graph)
                _, nodes_2, edges_2 = build_scene_graph(train_set.dataset.objects_dict, train_set.dataset.relationships_dict, data[0].output_graph)
                transf_node_1 = transform_locations(nodes_1, data.input_tf)
                transf_node_2 = transform_locations(nodes_2, data.input_tf)

        logger.log_validation_epoch()

    train_loss_plt = logger.plot_training_losses()
    val_loss_plt = logger.plot_valid_losses()
    acc_plt = logger.plot_accuracies()

    def _to_img(buf):
        import PIL
        from torchvision.transforms import ToTensor
        image = PIL.Image.open(buf)
        image = ToTensor()(image)
        return image

    writer.add_image("training loss", _to_img(train_loss_plt))
    writer.add_image("validation loss", _to_img(val_loss_plt))
    writer.add_image("accuracies", _to_img(acc_plt))


    for e, l in enumerate(logger.train_losses):
        writer.add_scalar("Loss/train", l, e)
    for e, l in enumerate(logger.val_losses):
        writer.add_scalar("Loss/eval", l, e)

    writer.flush()
    torch.save(neuralnet, f"{log_dir}/model_final.pt")
    stdout_contents = stdout_str.getvalue()
    with open(f"{log_dir}/stdout.log", 'w') as f:
        f.write(stdout_contents)
    with open(f"{log_dir}/training_logger.pickle", 'wb') as f:
        pickle.dump(logger, f)


@gin.configurable
def train_variability_main(
        test_split_file: str,
    ):

    dataset_cfg = DatasetCfg()
    dataset = SceneGraphChangeDataset()
    one_graph = dataset[0]
    num_node_attrs = one_graph.x.shape[-1]
    num_edge_attrs = one_graph.edge_attr.shape[-1]
    num_outputs = one_graph.y.shape[-1]
    stdout.warning(f"dataset configured to have {num_node_attrs=}, {num_edge_attrs=}, {num_outputs=}")
    test_split = np.load(test_split_file)
    tv_split = [val for val in range(len(dataset)) if val not in test_split]
    tv_set = dataset[tv_split]

    samples = torch.tensor([torch.sum(tv_set.data.state_mask), torch.sum(tv_set.data.y[:, 2]), torch.numel(tv_set.data.y[:, 2])])
    positives = torch.sum(tv_set.data.y, dim=0)
    cur_balance = torch.divide(positives, samples)
    stdout.warning(f"{cur_balance=}")

    hyperparams = {
        "model_name": "Simple GNN v1",
        "model_type": "gnn",
        "hidden_layers": [16],
        "lr": 0.0001,
        "weight_decay": 5e-4,
        "num_epochs": 20,
        "bs": 16,
        "loss": torch.nn.BCEWithLogitsLoss(reduction="none")
    }

    model = SimpleMPGNN(tv_set.num_node_features, tv_set.num_classes, tv_set.num_edge_features, [16])
    train_neuralnet(dset=tv_set, neuralnet=model, nnet_type="gnn", visualize=False)

    hyperparams = {
        "model_name": "Simple MLP Baseline",
        "model_type": "mlp",
        "hidden_layers": [32, 32],
        "lr": 0.01,
        "weight_decay": 5e-4,
        "num_epochs": 10,
        "bs": 16,
        "loss": torch.nn.BCEWithLogitsLoss(reduction="none")
    }

    model = SimpleMLP(tv_set.num_node_features, tv_set.num_classes + 1, hyperparams["hidden_layers"])
    train_neuralnet(tv_set, neuralnet=model, epochs=10, nnet_type="mlp", visualize=False)

if __name__ == "__main__":
    import sys
    FORMAT = "[%(asctime)s %(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    import io
    stdout_str = io.StringIO()
    ss = logging.StreamHandler(stdout_str)
    ss.setLevel(logging.DEBUG)
    logging.basicConfig(level='INFO', format=FORMAT)
    stdout.addHandler(ss)
    train_logger = logging.getLogger("train/eval")
    train_logger.addHandler(ss)
    gin_config_files = sys.argv[1:]

    stdout.info(f"using {gin_config_files=}")
    gin.parse_config_files_and_bindings(gin_config_files, "", skip_unknown=True)
    global_config_str = gin.config_str()
    stdout.info(global_config_str)
    train_variability_main()
    stdout_str.close()
