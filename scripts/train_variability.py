import torch
from torch_geometric.loader import DataLoader
from dataset.SceneGraphChangeDataset import SceneGraphChangeDataset, transform_locations
from dataset.utils.extract_data import build_scene_graph
from dataset.utils.logging import TrainingLogger
from dataset.DatasetCfg import DatasetCfg
from models.SimpleGCN import SimpleMPGNN
from models.FocalLoss import FocalLoss
from models.SimpleMLP import SimpleMLP
from eval import calculate_conf_mat
import numpy as np
from tqdm import tqdm
import os


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
    loss_tensor = l_fn(pred_i, data.y)
    loss_state = loss_tensor[:, 0]
    loss_pos = loss_tensor[:, 1]
    loss_mask = loss_tensor[:, 2]
    loss = (torch.sum(torch.multiply(loss_state, state_mask)) + torch.sum(torch.multiply(loss_pos, node_mask)) + torch.sum(loss_mask)) / (torch.sum(node_mask) + torch.sum(state_mask) + torch.numel(node_mask))
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

            state_valid = pred_i[torch.nonzero(data.state_mask == 1), 0]
            state_valid_label = data.y[torch.nonzero(data.state_mask == 1), 0]

            pos_valid = pred_i[torch.nonzero(data.y[:, 2] == 1), 1]
            pos_valid_label = data.y[torch.nonzero(data.y[:, 2] == 1), 1]

            node_valid = pred_i[:, 2]
            node_valid_label = data.y[:, 2]

            state_conf = calculate_conf_mat(state_valid, state_valid_label)
            pos_conf = calculate_conf_mat(pos_valid, pos_valid_label)
            node_conf = calculate_conf_mat(node_valid, node_valid_label)
            logger.log_validation_iter(val_loss, state_conf, pos_conf, node_conf)

            if visualize:
                _, nodes_1, edges_1 = build_scene_graph(train_set.dataset.objects_dict, train_set.dataset.relationships_dict, data[0].input_graph, "")
                _, nodes_2, edges_2 = build_scene_graph(train_set.dataset.objects_dict, train_set.dataset.relationships_dict, data[0].output_graph, "")
                transf_node_1 = transform_locations(nodes_1, data.input_tf)
                transf_node_2 = transform_locations(nodes_2, data.input_tf)

        logger.log_validation_epoch()

    logger.plot_training_losses()
    logger.plot_valid_losses()
    logger.plot_accuracies()


if __name__ == "__main__":
    dataset_cfg = DatasetCfg()
    dataset = SceneGraphChangeDataset(cfg=dataset_cfg)
    test_split_path = os.path.join(dataset_cfg.root, "results/ml_models/test_split.npy")
    test_split = np.load(test_split_path)
    tv_split = [val for val in range(len(dataset)) if val not in test_split]
    tv_set = dataset[tv_split]

    samples = torch.tensor([torch.sum(tv_set.data.state_mask), torch.sum(tv_set.data.y[:, 2]), torch.numel(tv_set.data.y[:, 2])])
    positives = torch.sum(tv_set.data.y, dim=0)
    cur_balance = torch.divide(positives, samples)
    pos_weights = torch.divide(samples, positives)
    sqrt_weights = torch.sqrt(pos_weights)

    hyperparams = {
        "model_name": "Simple GNN v3 focal loss",
        "model_type": "gnn",
        "hidden_layers": [16],
        "lr": 0.0001,
        "weight_decay": 5e-4,
        "num_epochs": 20,
        "bs": 16,
        "loss": FocalLoss(torch.ones((1, 3)), 0.5)
    }

    model = SimpleMPGNN(tv_set.num_node_features, tv_set.num_classes, tv_set.num_edge_features, hyperparams["hidden_layers"])
    train_neuralnet(tv_set, model, hyperparams, visualize=False)

    # hyperparams = {
    #     "model_name": "Simple MLP Baseline",
    #     "model_type": "mlp",
    #     "hidden_layers": [32, 32],
    #     "lr": 0.01,
    #     "weight_decay": 5e-4,
    #     "num_epochs": 10,
    #     "bs": 16,
    #     "loss": torch.nn.BCEWithLogitsLoss(reduction="none")
    # }
    #
    # model = SimpleMLP(tv_set.num_node_features, tv_set.num_classes + 1, hyperparams["hidden_layers"])
    # train_neuralnet(tv_set, model, hyperparams, visualize=False)
