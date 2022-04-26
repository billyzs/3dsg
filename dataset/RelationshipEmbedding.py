from collections import OrderedDict
from Relationships3DSSG import Relationships3DSSG
from DatasetCfg import DatasetCfg
import itertools
import logging
import torch
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


def get_relative_dist(node_1: Tuple, node_2: Tuple) -> torch.Tensor:
    return torch.Tensor(node_1[1]["attributes"]["location"] - node_2[1]["attributes"]["location"])


class BinaryEdgeEmbedding:
    def __init__(self, cfg: DatasetCfg.RelationshipParams):
        self.cfg: DatasetCfg.RelationshipParams = cfg
        self.loc: bool = cfg.relative_loc
        self.n: int = len(Relationships3DSSG)

    def generate_edge_embeddings(self, nodes_1: List[Tuple],
                                 edges: List[List],
                                 node_idx: List[int]) -> (torch.Tensor, torch.Tensor):
        node_local_id_to_idx = {int(node[0]): idx for (idx, node) in enumerate(nodes_1)}
        num_nodes = len(node_idx)
        num_edges = num_nodes * (num_nodes - 1)

        edge_map: Dict[Tuple[int, int], int] = {(node_local_id_to_idx[e[0]],\
            node_local_id_to_idx[e[1]]): e[2]-1 for e in edges}
        relative_loc = torch.zeros(num_edges, 3)
        _edge_idx_tensor = torch.zeros((2, num_edges))
        _edge_embeds_tensor = torch.zeros(num_edges, self.n)
        for (idx, (fr, to)) in enumerate(itertools.permutations(range(num_nodes), 2)):
            # (0,1), ...(0,9), (1,0), (1,2), ... (1,9), ...
            _edge_idx_tensor[0, idx] = fr
            _edge_idx_tensor[1, idx] = to
            relative_loc[idx, :] = get_relative_dist(nodes_1[fr], nodes_1[to]).flatten()
            rel = edge_map.get((fr, to), None)
            if rel is not None:
                _edge_embeds_tensor[idx, rel] = 1
        if self.loc:
            dist = torch.norm(relative_loc, p=2, dim=1)
            max_dist = torch.max(dist)
            dist = dist / max_dist
            relative_loc = relative_loc * dist.repeat(3,1).transpose(1,0)
            _edge_embeds_tensor = torch.hstack((_edge_embeds_tensor, relative_loc))
        return _edge_embeds_tensor, _edge_idx_tensor
