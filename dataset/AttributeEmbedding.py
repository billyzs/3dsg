from collections import OrderedDict
from .Attributes3DSSG import Attributes3DSSG
from .Taxonomy3DSSG import Objects3DSSG
import logging
from .DatasetCfg import DatasetCfg
import torch
from typing import List, Dict, Tuple
logger = logging.getLogger(__name__)


class BinaryNodeEmbedding:
    def __init__(self, att_cfg: DatasetCfg.AttributeParams, obj_cfg: DatasetCfg.ObjectParams):
        self.att_cfg: DatasetCfg.AttributeParams = att_cfg
        self.obj_cfg: DatasetCfg.ObjectParams = obj_cfg
        self.loc: bool = att_cfg.global_loc


    # Generates tensor of embeddings for a single graph sample
    def generate_node_embeddings(self, node_list: List[Tuple]) -> (torch.Tensor, List):
        node_dict = {node[0]: node[1] for node in node_list}
        node_ids = sorted(list(node_dict.keys()))
        num_nodes = len(node_ids)
        attribute_embeddings = torch.zeros((num_nodes, len(Attributes3DSSG)))
        node_locations = torch.zeros((num_nodes, 3))
        node_classifications = torch.zeros((num_nodes, 1))

        for idx, node in enumerate(node_ids):
            attribute_embeddings[idx, :] = self.calc_node_embedding(node_dict[node]["attributes"])
            node_locations[idx, :] = node_dict[node]["attributes"]["location"].flatten()
            node_classifications[idx] = int(node_dict[node]["global_id"])

        if self.loc:
            attribute_embeddings = torch.hstack([attribute_embeddings, node_locations])
        return attribute_embeddings, node_ids, node_locations, node_classifications

    def calc_node_embedding(self, node_dict: Dict) -> torch.Tensor:
        # Current embedding method: not embedding objects, embedding all attributes as binary variable
        raw_embedding: List[str] = []
        for (key, val) in node_dict.items():
            if key != "location":
                for v in val:
                    raw_embedding.append("_".join([key, v]))
        raw_embedding = [str(Attributes3DSSG.to_enum(r)) for r in raw_embedding]
        torch_embedding = torch.tensor(Attributes3DSSG.binary_encode(raw_embedding))
        return torch.unsqueeze(torch_embedding, 0)
