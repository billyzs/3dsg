from typing import List
from torch_geometric.data import Data
from Relationships3DSSG import Relationships3DSSG
import torch
import logging


logger = logging.getLogger(__name__)


class RelationshipsAllowList:
    def __init__(
        self,
        allowed_relationships: List[str],
    ):
        self.allow_list = allowed_relationships
        s = set(self.allow_list)  #TODO bzs convert str to enum
        self.allow_mask = [rel in s for rel in Relationships3DSSG]

    def remove_unused_relationships(self, rels: torch.Tensor) -> None:
        # TODO bzs actually test this
        rels = rels[:, self.allow_mask]


    def __call__(self, graph: Data) -> Data:
        self.remove_unused_relationships(graph.edge_attr)
        return graph
