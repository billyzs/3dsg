from collections import OrderedDict
from Attributes3DSSG import Attributes3DSSG
from Taxonomy3DSSG import Objects3DSSG
import logging
from dataset.DatasetCfg import DatasetCfg
import torch
from typing import List, Dict, Tuple
logger = logging.getLogger(__name__)


class BinaryNodeEmbedding:
    def __init__(self, att_cfg: DatasetCfg.AttributeParams, obj_cfg: DatasetCfg.ObjectParams):
        self.att_cfg: DatasetCfg.AttributeParams = att_cfg
        self.obj_cfg: DatasetCfg.ObjectParams = obj_cfg
        self.loc: bool = att_cfg.global_loc
        self.obj_mode: str = obj_cfg.mode
        self.allowlist: List[str] = att_cfg.allowed_attributes
        self.attributes: Dict[List] = self.build_attribute_set()
        self.objects: List[str] = self.build_object_set()

    # Creates a consistent ordered dictionary of attributes used for binary embedding
    def build_attribute_set(self) -> Dict:
        if self.allowlist is None:
            raise ValueError("no attributes specified to build node embedding")
        logger.info(f"processing {self.allowlist=}")
        attributes = OrderedDict()  # import to keep order for consistency

        # will raise if input invalid
        attr_enum = {Attributes3DSSG.to_enum(a): a for a in self.allowlist}
        enums = sorted(list(attr_enum.keys()))
        for e in enums:
            attr = attr_enum[e]
            category, val = attr.split(":")
            attributes.setdefault(category, []).append(val)
        logger.info(f"allowing the following attributes: {attributes}")
        return attributes

    # Creates a consistent ordered dictionary of attributes used for binary embedding
    def build_object_set(self) -> List[str]:
        if self.obj_mode == "full":
            object_set = [obj.name for obj in Objects3DSSG]
        else:
            raise Exception("Currently not supporting any other object mode")

        return object_set

    # Generates tensor of embeddings for a single graph sample
    def generate_node_embeddings(self, node_list: List[Tuple]) -> (torch.Tensor, List):
        node_embeddings = []
        node_ids = []
        for node in node_list:
            node_embedding = self.calc_node_embedding(node[1]["attributes"])
            node_embeddings.append(node_embedding)
            node_ids.append(node[0])

        return torch.cat(node_embeddings, 0), node_ids

    def calc_node_embedding(self, node_dict: Dict) -> torch.Tensor:
        # Current embedding method: not embedding objects, embedding all attributes as binary variable
        embedding = []
        for att_type in self.attributes.keys():
            node_embed = [0] * len(self.attributes[att_type])
            if att_type in node_dict.keys():
                for att in node_dict[att_type]:
                    node_embed[self.attributes[att_type].index(att)] = 1
            embedding += node_embed

        if self.loc:
            torch_embedding = torch.cat((torch.Tensor(embedding), torch.Tensor(node_dict["location"])[:, 0]), 0)
        else:
            torch_embedding = torch.Tensor(embedding)

        return torch.unsqueeze(torch_embedding, 0)
