import logging
import os
import json
import torch
import uuid
from typing import List, Dict
from torch_geometric.data import InMemoryDataset, Data
import numpy as np
from tqdm import tqdm
import gin
from extract_data import build_scene_graph, format_scan_dict
from AttributesAllowList import AttributesAllowList


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


class FilteredSceneGraphChangeDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.uuid = uuid.uuid4()
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
        else:
            return []

    @property
    def processed_file_names(self):
        return [f'scene_graph_data_{self.uuid}.pt',
                f"{self.uuid}.txt",
                ]

    def download(self):
        pass
        # raise Exception("Files Not Found!")

    def parse_attributes(self):
        with open(self.attributes_file) as f:
            atts = f.readlines()
            for att in atts:
                att_class, att_value = att.rstrip("\n").split(":")
                if att_class not in self.attributes_dict.keys():
                    self.attributes_dict[att_class] = [att_value]
                else:
                    self.attributes_dict[att_class].append(att_value)

        #del self.attributes_dict["other"]
        #del self.attributes_dict["symmetry"]
        #del self.attributes_dict["style"]

    def calc_node_embedding(self, node_dict):
        embedding = []
        for att_type in self.attributes_dict.keys():
            node_embed = [0] * len(self.attributes_dict[att_type])
            if att_type in node_dict.keys():
                for att in node_dict[att_type]:
                    node_embed[self.attributes_dict[att_type].index(att)] = 1
            embedding += node_embed

        torch_embedding = torch.cat((torch.Tensor(embedding), torch.Tensor(node_dict["location"])), 0)
        return torch.unsqueeze(torch_embedding, 0)

    def format_node_embeddings(self, node_list):
        node_embeddings = []
        node_ids = []
        for node in node_list:
            node_embedding = self.calc_node_embedding(node[1]["attributes"])
            node_embeddings.append(node_embedding)
            node_ids.append(node[0])

        return torch.cat(node_embeddings, 0), node_ids

    def format_node_labels(self, node_list, input_node_idx):
        # TODO: Implement label generation
        output_node_idx = [val[0] for val in node_list]
        output_nodes = [val[1] for val in node_list]
        output_embeddings = []
        output_mask = torch.ones(len(input_node_idx))
        for i in range(len(input_node_idx)):
            idx = input_node_idx[i]
            if idx in output_node_idx:
                output_node = output_nodes[output_node_idx.index(idx)]
                assert node_list[output_node_idx.index(idx)][0] == idx, "You fucked up"
                output_embedding = self.format_label_embeddings(output_node["attributes"])
                output_embeddings.append(output_embedding)
            else:

                output_embeddings.append(torch.zeros((1, self.label_len)))
                output_mask[i] = 0

        output_embeddings_tensor = torch.cat(output_embeddings)
        return output_embeddings_tensor, output_mask

    def format_label_embeddings(self, node_dict):
        state_embed = [0] * len(self.attributes_dict["state"])
        if "state" in node_dict.keys():
            for att in node_dict["state"]:
                state_embed[self.attributes_dict["state"].index(att)] = 1

        torch_embedding = torch.cat((torch.Tensor(state_embed), torch.Tensor(node_dict["location"])), 0)
        return torch.unsqueeze(torch_embedding, 0)

    def format_edge_embeddings(self, edges, node_idx):
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
                        _, nodes_1, edges_1 = build_scene_graph(self.objects_dict, self.relationships_dict, scan_id_set[i])
                        _, nodes_2, edges_2 = build_scene_graph(self.objects_dict, self.relationships_dict, scan_id_set[j])
                        if nodes_1 is not None and nodes_2 is not None:
                            node_embeddings, node_idxs = self.format_node_embeddings(nodes_1)
                            node_labels, node_masks = self.format_node_labels(nodes_2, node_idxs)
                            edge_idx, edge_embeddings = self.format_edge_embeddings(edges_1, node_idxs)
                            sample = Data(x=node_embeddings, edge_index=edge_idx, edge_attr=edge_embeddings, y=node_labels, mask=node_masks)
                            samples.append(sample)

        data, slices = self.collate(samples)
        torch.save((data, slices), self.processed_paths[0])
        config_file_name = self.processed_paths[1]
        with open(config_file_name, 'w') as f:
            print(self.attributes_dict, file=f)





@gin.configurable
def make_dataset(
        dataset_root: str,
        allowed_attributes: List[str],
    ):
    logger.debug(f"{dataset_root=}")
    allow_list = AttributesAllowList(allowed_attributes)
    dataset = FilteredSceneGraphChangeDataset(
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
    #  gin.parse_config_files_and_bindings("config.gin")
    gin.parse_config_file("config.gin")
    d = dataset_main()
