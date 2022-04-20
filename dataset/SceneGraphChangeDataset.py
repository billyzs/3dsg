import json
import logging
import os
from typing import List
import uuid
import torch
from torch_geometric.data import InMemoryDataset, Data
from utils.extract_data import build_scene_graph, format_scan_dict, transform_locations
from dataset import AttributesAllowList
import numpy as np
from tqdm import tqdm


logger = logging.getLogger(__name__)


def get_scene_list(scene):
    scan_id_set = [scene["reference"]]
    scan_tf_set = [np.eye(4)]
    for follow_scan in scene["scans"]:
        scan_id_set.append(follow_scan["reference"])
        if "transform" in follow_scan.keys():
            scan_tf_set.append(np.asarray(follow_scan["transform"]).reshape((4, 4)).T)
        else:
            scan_tf_set.append(np.eye(4))

    return scan_id_set, scan_tf_set


def get_relative_dist(node_1, node_2):
    return node_1[1]["attributes"]["location"] - node_2[1]["attributes"]["location"]


class SceneGraphChangeDataset(InMemoryDataset):
    def __init__(self, root, loc_mode="rel", label_mode="thresh", threshold=0.1, transform=None, pre_transform=None, pre_filter=None):
        # Location representation mode
        #   "rel": Relative positions as graph node relationships,
        #   "global": Global position vectors as graph node attributes)
        self.uuid = uuid.uuid4()
        self.loc_mode = loc_mode

        # Location label mode:
        #   "thresh": labelled as binary classification: stationary (0) if movement between scenes below a certain
        #   threshold, or in motion (1) if above threshold)
        #   "dist": learn a normally distributed PDF of location of second object centered around first object
        self.label_mode = label_mode
        self.threshold = threshold

        self.raw_files = os.path.join(root, "raw", "raw_files.txt")
        self.scans = json.load(open(os.path.join(root, "raw", "3RScan.json")))
        objects_json = json.load(open(os.path.join(root, "raw", "scene-graphs", "objects.json")))
        relationships_json = json.load(open(os.path.join(root, "raw", "scene-graphs", "relationships.json")))
        self.attributes_file = os.path.join(root, "raw", "scene-graphs", "attributes.txt")
        self.attributes_dict = {}
        # self.parse_attributes()
        if isinstance(pre_transform, AttributesAllowList):
            self.attributes_dict = pre_transform.attributes
        self.label_len = len(self.attributes_dict["state"]) + 3

        self.objects_dict = format_scan_dict(objects_json, "objects")
        self.relationships_dict = format_scan_dict(relationships_json, "relationships")
        with open(os.path.join(root, "raw", "scene-graphs", "relationships.txt")) as f:
            lines = f.readlines()
        self.num_relationships = len(lines)

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        if os.path.isfile(self.raw_files):
            with open(self.raw_files) as f:
                files = f.read().splitlines()
        return files

    @property
    def processed_file_names(self):
        return [f"scene_graph_data_{self.uuid}.pt", f"{self.uuid}.txt"]

    def download(self):
        pass
        #raise Exception("Files Not Found!")

    def parse_attributes(self):
        with open(self.attributes_file) as f:
            atts = f.readlines()
            for att in atts:
                att_class, att_value = att.rstrip("\n").split(":")
                if att_class not in self.attributes_dict.keys():
                    self.attributes_dict[att_class] = [att_value]
                else:
                    self.attributes_dict[att_class].append(att_value)

        # del self.attributes_dict["other"]
        # del self.attributes_dict["symmetry"]
        # del self.attributes_dict["style"]

    def calc_node_embedding(self, node_dict):
        embedding = []
        for att_type in self.attributes_dict.keys():
            node_embed = [0] * len(self.attributes_dict[att_type])
            if att_type in node_dict.keys():
                for att in node_dict[att_type]:
                    node_embed[self.attributes_dict[att_type].index(att)] = 1
            embedding += node_embed

        if self.loc_mode == "global":
            torch_embedding = torch.cat((torch.Tensor(embedding), torch.Tensor(node_dict["location"])[:, 0]), 0)
        else:
            torch_embedding = torch.Tensor(embedding)

        return torch.unsqueeze(torch_embedding, 0)

    def format_node_embeddings(self, node_list):
        node_embeddings = []
        node_ids = []
        for node in node_list:
            node_embedding = self.calc_node_embedding(node[1]["attributes"])
            node_embeddings.append(node_embedding)
            node_ids.append(node[0])

        return torch.cat(node_embeddings, 0), node_ids

    def format_node_labels(self, nodes_1, nodes_2, input_node_idx):
        output_node_idx = [val[0] for val in nodes_2]
        output_nodes = [val for val in nodes_2]
        input_nodes = [val for val in nodes_1]
        output_embeddings = []
        output_mask = torch.ones(len(input_node_idx))
        for i in range(len(input_node_idx)):
            idx = input_node_idx[i]
            if idx in output_node_idx:
                output_node = output_nodes[output_node_idx.index(idx)]
                input_node = input_nodes[i]
                output_embedding = self.get_variability_embedding(input_node, output_node)
                output_embeddings.append(output_embedding)
            else:
                output_embeddings.append(torch.zeros((1, 2)))
                output_mask[i] = 0

        output_embeddings_tensor = torch.cat(output_embeddings)
        return output_embeddings_tensor, output_mask

    def get_variability_embedding(self, in_node, out_node):
        state_diff = 0
        if "state" in in_node[1]["attributes"].keys() and "state" in out_node[1]["attributes"].keys():
            state_diff = int(in_node[1]["attributes"]["state"] != out_node[1]["attributes"]["state"])

        dist = np.linalg.norm(get_relative_dist(in_node, out_node))
        pos_diff = int(dist > self.threshold)
        return torch.Tensor([[state_diff, pos_diff]])

    def format_edge_embeddings(self, nodes_1, edges, node_idx):
        edge_idx = []
        edge_embeds = []
        for edge in edges:
            n_1 = node_idx.index(edge[0])
            n_2 = node_idx.index(edge[1])
            rel = edge[2] - 1
            if (n_1, n_2) in edge_idx:
                idx = edge_idx.index((n_1, n_2))
                edge_embeds[idx][rel] = 1
            else:
                edge_idx.append((n_1, n_2))
                edge_embed = [0] * self.num_relationships
                edge_embed[rel] = 1
                if self.loc_mode == "rel":
                    edge_embed += get_relative_dist(nodes_1[n_1], nodes_1[n_2]).flatten().tolist()
                edge_embeds.append(edge_embed)

        edge_idx_tensor = torch.transpose(torch.Tensor(edge_idx), 0, 1)
        edge_embeds_tensor = torch.Tensor(edge_embeds)
        return edge_idx_tensor, edge_embeds_tensor

    def process(self):
        samples = []
        for scene in tqdm(self.scans):
            scan_id_set, scan_tf_set = get_scene_list(scene)
            for i in range(len(scan_id_set)):
                for j in range(len(scan_id_set)):
                    if i != j:
                        _, nodes_1, edges_1 = build_scene_graph(self.objects_dict, self.relationships_dict, scan_id_set[i], self.root)
                        _, nodes_2, edges_2 = build_scene_graph(self.objects_dict, self.relationships_dict, scan_id_set[j], self.root)
                        T_1I = scan_tf_set[i]
                        T_2I = scan_tf_set[j]
                        if nodes_1 is not None and nodes_2 is not None:
                            transf_node_1 = transform_locations(nodes_1, T_1I)
                            transf_node_2 = transform_locations(nodes_2, T_2I)
                            node_embeddings, node_idxs = self.format_node_embeddings(transf_node_1)
                            node_labels, node_masks = self.format_node_labels(transf_node_1, transf_node_2, node_idxs)
                            edge_idx, edge_embeddings = self.format_edge_embeddings(transf_node_1, edges_1, node_idxs)
                            sample = Data(
                                x=node_embeddings,
                                edge_index=edge_idx,
                                edge_attr=edge_embeddings,
                                y=node_labels,
                                mask=node_masks,
                                input_graph=scan_id_set[i],
                                output_graph=scan_id_set[j],
                                input_tf=T_1I,
                                output_tf=T_2I
                            )
                            samples.append(sample)

        data, slices = self.collate(samples)
        torch.save((data, slices), self.processed_paths[0])


@gin.configurable
def make_dataset(
        dataset_root: str,
        allowed_attributes: List[str],
    ):
    logger.debug(f"{dataset_root=}")
    allow_list = AttributesAllowList(allowed_attributes)
    dataset = SceneGraphChangeDataset(
        dataset_root,
        pre_transform=allow_list,
    )
    return dataset


@gin.configurable
def dataset_main(
        log_level: str,
    ):
    logging.basicConfig(level=log_level)
    dataset = make_dataset()
    return dataset


if __name__ == "__main__":
    gin.parse_config_file("config.gin")
    d = dataset_main()
    
    data_folder = "/home/sam/ethz/plr/plr-2022-predicting-changes/data"
    dataset = SceneGraphChangeDataset(data_folder, loc_mode="rel")

    # Calculate Class Imbalance:
    state_var = [0, 0]
    pos_var = [0, 0]
    node_var = [0, 0]
    for i in range(len(dataset)):
        data = dataset[i]
        state_var[0] += torch.sum(data.y[:, 0])
        pos_var[0] += torch.sum(data.y[:, 1])
        node_var[0] += torch.sum(data.mask)

        state_var[1] += torch.numel(data.y[:, 0])
        pos_var[1] += torch.numel(data.y[:, 1])
        node_var[1] += torch.numel(data.mask)

    print("State Variability: {}/{}".format(state_var[0], state_var[1]))
    print("Position Variability: {}/{}".format(pos_var[0], pos_var[1]))
    print("Node Variability: {}/{}".format(node_var[0], node_var[1]))


