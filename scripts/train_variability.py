import torch
from torch_geometric.loader import DataLoader
from dataset.SceneGraphChangeDataset import SceneGraphChangeDataset, transform_locations
from dataset.utils.extract_data import build_scene_graph
from dataset.utils.logging import TrainingLogger
from dataset.DatasetCfg import DatasetCfg
from models.SimpleGCN import SimpleMPGNN
from models.SimpleMLP import SimpleMLP
from eval import calculate_conf_mat
import numpy as np
from tqdm import tqdm


def calculate_training_loss(data, nnet, l_fn, nnet_type):
    x_i = data.x
    y_i = data.y[:, :2]
    m_i = data.y[:, 2]
    if nnet_type == "gnn":
        e_idx_i = data.edge_index.type(torch.LongTensor)
        e_att_i = data.edge_attr
        pred_i = nnet(x_i, e_idx_i, e_att_i)
    else:
        pred_i = nnet(x_i)
    loss_1 = l_fn(pred_i[:, :2], y_i)
    loss_2 = l_fn(pred_i[:, 2], m_i)
    loss = (torch.sum(torch.multiply(torch.sum(loss_1, 1), m_i))+ torch.sum(loss_2))/(2*torch.sum(m_i) + torch.numel(loss_2))
    return loss, pred_i


def train_neuralnet(dset, neuralnet, hyperparams, visualize=False):
    train_n = int(len(dset) * 0.85)
    train_set, val_set = torch.utils.data.random_split(dset,  [train_n, len(dset)-train_n])
    train_loader = DataLoader(train_set, batch_size=hyperparams["bs"])
    val_loader = DataLoader(val_set, batch_size=1)
    l_fn = hyperparams["loss"]
    epochs = hyperparams["num_epochs"]
    nnet_type = hyperparams["model_type"]
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["weight_decay"])

    logger = TrainingLogger(hyperparams["model_name"], print_interval=20, train_n=int(train_n/hyperparams["bs"]))
    for i in range(epochs):
        print("Training. Epoch {}".format(i+1))
        neuralnet.train()
        for data in train_loader:
            optimizer.zero_grad()
            loss, _ = calculate_training_loss(data, neuralnet, l_fn, nnet_type)
            logger.log_training_iter(loss.item())
            loss.backward()
            optimizer.step()
        logger.log_training_epoch(model)

        print("Validation:")
        model.eval()
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

    logger.plot_training_losses()
    logger.plot_valid_losses()
    logger.plot_accuracies()


if __name__ == "__main__":
    dataset_cfg = DatasetCfg()
    dataset = SceneGraphChangeDataset(cfg=dataset_cfg)
    test_split = np.load("/home/sam/ethz/plr/plr-2022-predicting-changes/data/results/ml_models/test_split.npy")
    tv_split = [val for val in range(len(dataset)) if val not in test_split]
    tv_set = dataset[tv_split]

    hyperparams = {
        "model_name": "Simple GNN v1",
        "model_type": "gnn",
        "hidden_layers": [16],
        "lr": 0.001,
        "weight_decay": 5e-4,
        "num_epochs": 10,
        "bs": 16,
        "loss": torch.nn.BCEWithLogitsLoss(reduction="none")
    }

    model = SimpleMPGNN(tv_set.num_node_features, tv_set.num_classes, tv_set.num_edge_features, hyperparams["hidden_layers"])
    train_neuralnet(tv_set, model, hyperparams, visualize=False)

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
    train_neuralnet(tv_set, model, hyperparams, visualize=False)