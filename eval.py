import torch
from torch_geometric.loader import DataLoader
from dataset.SceneGraphChangeDataset import SceneGraphChangeDataset, transform_locations
from utils.logging import EvalLogger
from models.SimpleGCN import SimpleMPGNN
from models.SimpleMLP import SimpleMLP
import numpy as np
from tqdm import tqdm


def calculate_conf_mat(pred, label):
    class_preds = (pred > 0).float()
    tp = torch.sum(torch.minimum(class_preds, label))
    tn = label.shape[0] - torch.sum(torch.maximum(class_preds, label))
    fp = torch.sum(class_preds) - tp
    fn = torch.sum(label) - tp
    return torch.tensor([[tp, fp], [fn, tn]])


def eval(dataset, model, nnet_type, title):
    logger = EvalLogger(title)
    for data in tqdm(dataset):
        if nnet_type == "gnn":
            e_idx_i = data.edge_index.type(torch.LongTensor)
            e_att_i = data.edge_attr
            pred_i = model(data.x, e_idx_i, e_att_i)
        else:
            pred_i = model(data.x)
        state_conf = calculate_conf_mat(pred_i[:, 0], data.y[:, 0])
        pos_conf = calculate_conf_mat(pred_i[:, 1], data.y[:, 1])
        mask_conf = calculate_conf_mat(pred_i[:, 2], data.mask)

        logger.log_eval_iter(state_conf, pos_conf, mask_conf)

    logger.print_eval_results()


if __name__ == "__main__":
    data_folder = "/home/sam/ethz/plr/plr-2022-predicting-changes/data"
    dataset = SceneGraphChangeDataset(data_folder, loc_mode="rel", label_mode="thresh", threshold=0.5)
    test_split = np.load("test_split.npy")
    test_set = dataset[test_split]

    model_path = "simple_gnn_v1.pt"
    cur_model_hidden = [16]
    gnn = SimpleMPGNN(test_set.num_node_features, test_set.num_classes + 1, test_set.num_edge_features, cur_model_hidden)
    gnn.load_state_dict(torch.load(model_path))
    gnn.eval()

    eval(test_set, gnn, "gnn", "Simple GNN v1")

    model_path = "simple_mlp_baseline.pt"
    cur_model_hidden = [32, 32]
    mlp = SimpleMLP(test_set.num_node_features, test_set.num_classes + 1, cur_model_hidden)
    mlp.load_state_dict(torch.load(model_path))
    mlp.eval()

    eval(test_set, mlp, "mlp", "Simple MLP Baseline")

