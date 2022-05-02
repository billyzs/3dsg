import torch
from torch.utils.data import DataLoader
from dataset.utils.logging import EvalLogger
from dataset.DatasetCfg import DatasetCfg
from dataset.SceneGraphChangeDataset import SceneGraphChangeDataset
from models.SimpleGCN import SimpleMPGNN
from models.SimpleMLP import SimpleMLP
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt


def calculate_conf_mat(pred, label):
    class_preds = (pred > 0).float()
    tp = torch.sum(torch.minimum(class_preds, label))
    tn = label.shape[0] - torch.sum(torch.maximum(class_preds, label))
    fp = torch.sum(class_preds) - tp
    fn = torch.sum(label) - tp
    return torch.tensor([[tp, fp], [fn, tn]])


def eval_var(dataset, model, nnet_type, title):
    logger = EvalLogger(title)
    for data in tqdm(dataset):
        if nnet_type == "gnn":
            e_idx_i = data.edge_index.type(torch.LongTensor)
            e_att_i = data.edge_attr
            pred_i = model(data.x, e_idx_i, e_att_i)
        else:
            pred_i = model(data.x)

        valid_samples = pred_i[torch.where(data.y[:, 2] == 1)]
        valid_sample_label = data.y[torch.where(data.y[:, 2] == 1)]
        state_conf = calculate_conf_mat(valid_samples[:, 0], valid_sample_label[:, 0])
        pos_conf = calculate_conf_mat(valid_samples[:, 1], valid_sample_label[:, 1])
        mask_conf = calculate_conf_mat(pred_i[:, 2], data.y[:, 2])

        logger.log_eval_iter(state_conf, pos_conf, mask_conf)

    logger.print_eval_results()


def plot_reconstruction(x_in, x_reconstructed):
    n = x_in.shape[0]
    y = x_reconstructed.tolist()
    x = [i for i in range(n)]
    colors = ["b" if x_in[i].item() == 0 else "r" for i in range(n)]
    plt.bar(x, y, color=colors)
    plt.show()


def plot_sparse_rep(x):
    plt.imshow(torch.reshape(x, (27, 23)).detach().numpy())
    plt.show()


def plot_embedding_vec(x):
    plt.imshow(torch.reshape(x, (32, 16)).detach().numpy())
    plt.show()


if __name__ == "__main__":
    dataset_cfg = DatasetCfg()
    dataset = SceneGraphChangeDataset(cfg=dataset_cfg)
    test_split = np.load("/home/sam/ethz/plr/plr-2022-predicting-changes/data/results/ml_models/test_split.npy")
    test_idx = [val for val in range(len(dataset)) if val in test_split]
    test_set = dataset[test_idx]

    model_path = "/home/sam/ethz/plr/plr-2022-predicting-changes/simple_gnn_v1.pt"
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

    model = SimpleMPGNN(test_set.num_node_features, test_set.num_classes, test_set.num_edge_features,
                        hyperparams["hidden_layers"])
    model.load_state_dict(torch.load(model_path))
    model.eval()

    eval_var(test_set, model, "gnn", hyperparams["model_name"])

