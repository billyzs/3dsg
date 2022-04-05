import torch
from torch_geometric.loader import DataLoader
from dataset.SceneGraphChangeDataset import SceneGraphChangeDataset
from models.SimpleGCN import SimpleMPGNN
import numpy as np
from tqdm import tqdm


def train_gnn(dset, gnn):
    train_n = int(len(dset) * 0.85)
    train_set, val_set = torch.utils.data.random_split(dset,  [train_n, len(dset)-train_n])
    # train_loader = DataLoader(train_set, batch_size=16)
    # val_loader = DataLoader(val_set, batch_size=16)
    gnn.train()
    l_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    epochs = 10
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for i in range(epochs):
        epoch_loss = 0
        for data in tqdm(train_set):
            optimizer.zero_grad()
            x_i = data.x
            y_i = data.y
            m_i = data.mask
            e_idx_i = data.edge_index.type(torch.LongTensor)
            e_att_i = data.edge_attr
            pred_i = gnn(x_i, e_idx_i, e_att_i)
            loss_1 = l_fn(pred_i[:, :2], y_i)
            loss_2 = l_fn(pred_i[:, 2], m_i)
            loss = torch.mean(torch.multiply(torch.sum(loss_1, 1), m_i)) + torch.mean(loss_2)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        print("Epoch {}. Loss = {}".format(i, epoch_loss))


if __name__ == "__main__":
    data_folder = "/home/sam/ethz/plr/plr-2022-predicting-changes/data"
    dataset = SceneGraphChangeDataset(data_folder, loc_mode="rel", label_mode="thresh", threshold=0.5)
    test_split = np.load("test_split.npy")
    tv_split = [val for val in range(len(dataset)) if val not in test_split]
    tv_set = dataset[tv_split]
    model = SimpleMPGNN(tv_set.num_node_features, tv_set.num_classes + 1, tv_set.num_edge_features)
    train_gnn(tv_set, model)


    # train_set, val_set, test_set = torch.utils.data.random_split(dataset, [0.75, 0.15, 0.1])