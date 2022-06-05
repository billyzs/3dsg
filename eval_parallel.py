import torch
from torch.utils.data import DataLoader
from dataset.utils.logging import EvalLogger
from dataset.DatasetCfg import DatasetCfg
from dataset.SceneGraphChangeDataset import SceneGraphChangeDataset
from models.SimpleGCN import SimpleMPGNN
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import logging
import gin
import os.path as osp
from train_variability_ablation import train_neuralnet, train_variability_main  # so that gin can pickup their params

stdout = logging.getLogger("eval_parallel")


def calculate_conf_mat(pred, label):
    class_preds = (pred > 0).float()
    tp = torch.sum(torch.minimum(class_preds, label))
    tn = label.shape[0] - torch.sum(torch.maximum(class_preds, label))
    fp = torch.sum(class_preds) - tp
    fn = torch.sum(label) - tp
    return torch.tensor([[tp, fp], [fn, tn]])


@torch.inference_mode()
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



def eval_main(test_split_file="",
              base_log_dir="",
              experiment_name="",
              **kwargs
              ):
    # dataset_cfg = DatasetCfg()
    dataset = SceneGraphChangeDataset()
    # test_split = np.load("/home/sam/ethz/plr/plr-2022-predicting-changes/data/results/ml_models/test_split.npy")
    _test_split = np.load(test_split_file)
    test_idx = [val for val in range(len(dataset)) if val in _test_split]
    test_set = dataset[test_idx]

    model_path = osp.join(base_log_dir, experiment_name, "gnn", "model_final.pt")
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

    _model = torch.load(model_path)
    model = SimpleMPGNN(test_set.num_node_features, test_set.num_classes, test_set.num_edge_features,
                        hyperparams["hidden_layers"])
    model.load_state_dict(_model.state_dict())
    del _model
    model.eval()

    stdout.info("loaded model")
    return
    eval_var(test_set, model, "gnn", experiment_name)

if __name__ == "__main__":
    import sys
    gin_config_files = sys.argv[1:]
    gin.parse_config_files_and_bindings(gin_config_files, "", skip_unknown=True)
    FORMAT = "[%(asctime)s %(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    import io
    stdout_str = io.StringIO()
    ss = logging.StreamHandler(stdout_str)
    ss.setLevel(logging.DEBUG)
    logging.basicConfig(level='INFO', format=FORMAT)
    stdout.addHandler(ss)
    train_logger = logging.getLogger("train/eval")
    train_logger.addHandler(ss)

    stdout.info(f"using {gin_config_files=}")
    global_config_str = gin.config_str()
    stdout.info(global_config_str)
    args = gin.get_bindings("train_neuralnet")
    args.update(gin.get_bindings("train_variability_main"))
    stdout.info(args)
    eval_main(**args)
    stdout_str.close()
