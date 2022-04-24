from collections import OrderedDict
from Relationships3DSSG import Relationships3DSSG
from dataset.DatasetCfg import DatasetCfg
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
        self.attributes: List[str] = self.validate_relationship_set(cfg.allowed_relationships)
        self.n: int = len(self.attributes)

    @staticmethod
    def validate_relationship_set(allow_list: List[str]) -> List[str]:
        if allow_list is None:
            raise ValueError("no attributes specified to build node embedding")
        logger.info(f"processing {allow_list=}")
        relationships = []

        rel_enum = {Relationships3DSSG.to_enum(a): a for a in allow_list}
        enums = sorted(list(rel_enum.keys()))
        for e in enums:
            rel = rel_enum[e]
            relationships.append(rel)
        logger.info(f"allowing the following attributes: {relationships}")
        return relationships

    def generate_edge_embeddings(self, nodes_1: List[Tuple], edges: List[List], node_idx: List[int]) -> (torch.Tensor, torch.Tensor):
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
                edge_embed = [0] * self.n
                edge_embed[rel] = 1
                if self.loc:
                    edge_embed += get_relative_dist(nodes_1[n_1], nodes_1[n_2]).flatten().tolist()
                edge_embeds.append(edge_embed)

        edge_idx_tensor = torch.transpose(torch.Tensor(edge_idx), 0, 1)
        edge_embeds_tensor = torch.Tensor(edge_embeds)
        return edge_embeds_tensor, edge_idx_tensor
