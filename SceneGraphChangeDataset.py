import torch
import os
from torch_geometric.data import InMemoryDataset
import json
from extract_data import build_scene_graph, format_scan_dict
import numpy as np


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


class SceneGraphChangeDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.raw_files = os.path.join(root, "raw", "raw_files.txt")
        self.scans = json.load(open(os.path.join(root, "raw", "3RScan.json")))
        objects_json = json.load(open(os.path.join(root, "raw", "scene-graphs", "objects.json")))
        relationships_json = json.load(open(os.path.join(data_folder, "raw", "scene-graphs", "relationships.json")))
        self.attributes_file = os.path.join(root, "raw", "scene-graphs", "attributes.txt")
        self.attributes_dict = {}
        self.parse_attributes()

        self.objects_dict = format_scan_dict(objects_json, "objects")
        self.relationships_dict = format_scan_dict(relationships_json, "relationships")

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
        return ['scene_graph_data.pt']

    def download(self):
        raise Exception("Files Not Found!")

    def parse_attributes(self):
        with open(self.attributes_file) as f:
            atts = f.readlines()
            for att in atts:
                att_class, att_value = att.rstrip("\n").split(":")
                if att_class not in self.attributes_dict.keys():
                    self.attributes_dict[att_class] = [att_value]
                else:
                    self.attributes_dict[att_class].append(att_value)

        del self.attributes_dict["other"]
        del self.attributes_dict["symmetry"]
        del self.attributes_dict["style"]

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
        pass

    def process(self):
        samples = []
        for scene in self.scans:
            scan_id_set, scan_tf_set = get_scene_list(scene)
            for i in range(len(scan_id_set)):
                for j in range(len(scan_id_set)):
                    if i != j:
                        _, nodes_1, edges_1 = build_scene_graph(self.objects_dict, self.relationships_dict, scan_id_set[i])
                        _, nodes_2, edges_2 = build_scene_graph(self.objects_dict, self.relationships_dict, scan_id_set[j])
                        if nodes_1 is not None and nodes_2 is not None:
                            node_embeddings, node_idxs = self.format_node_embeddings(nodes_1)
                            # TODO: label generation, mask generation, edge_generation,
                            #  and formatting into the PyTorch Dataset
                            # node_labels, node_masks = self.format_node_labels(nodes_2, node_idxs)







if __name__ == "__main__":
    data_folder = "/home/sam/ethz/plr/plr-2022-predicting-changes/data"
    dataset = SceneGraphChangeDataset(data_folder)