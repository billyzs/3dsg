from DatasetCfg import DatasetCfg
import logging
import torch
from typing import List, Dict, Tuple
logger = logging.getLogger(__name__)


def get_relative_dist(node_1: Tuple, node_2: Tuple) -> torch.Tensor:
    return torch.Tensor(node_1[1]["attributes"]["location"] - node_2[1]["attributes"]["location"])


class BinaryVariabilityEmbedding:
    def __init__(self, cfg: DatasetCfg.VariabilityParams):
        self.cfg: DatasetCfg.VariabilityParams = cfg
        self.threshold: float = self.cfg.threshold

    def generate_variability_embedding(self, nodes_1: List[Tuple], nodes_2: List[Tuple], input_node_idx: List[int]) -> torch.Tensor:
        output_node_idx = [val[0] for val in nodes_2]
        output_nodes = [val for val in nodes_2]
        input_nodes = [val for val in nodes_1]
        output_embeddings = []
        for i in range(len(input_node_idx)):
            idx = input_node_idx[i]
            if idx in output_node_idx:
                output_node = output_nodes[output_node_idx.index(idx)]
                input_node = input_nodes[i]
                var_embed = self.get_variability_embedding(input_node, output_node)
                output_embeddings.append(var_embed)
            else:
                output_embeddings.append(torch.zeros((1, 3)))

        output_embeddings_tensor = torch.cat(output_embeddings)
        return output_embeddings_tensor

    def get_variability_embedding(self, in_node: Tuple, out_node: Tuple) -> torch.Tensor:
        state_diff = 0
        if "state" in in_node[1]["attributes"].keys() and "state" in out_node[1]["attributes"].keys():
            state_diff = int(in_node[1]["attributes"]["state"] != out_node[1]["attributes"]["state"])

        dist = torch.norm(get_relative_dist(in_node, out_node))
        pos_diff = int(dist > self.threshold)
        return torch.Tensor([[state_diff, pos_diff, 1]])
