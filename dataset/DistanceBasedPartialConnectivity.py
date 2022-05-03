from typing import List
from collections.abc import Sequence, Callable
from dataclasses import dataclass
from torch_geometric.data import Data
from .Relationships3DSSG import Relationships3DSSG
import torch
import logging
import gin


logger = logging.getLogger(__name__)


@gin.configurable
@dataclass(frozen=True)
class DistanceBasedPartialConnectivity:
    """
    given a (normalized to 0 and 1) distance threshold, remove edges that (do not have any relations
    and have relative distance smaller than this threshold)
    this should be done before any attribute selection
    """
    distance_threshold: float
    enabled: bool = True
    def __call__(self, graph: Data) -> Data:
        return self.filter(graph) if self.enabled else graph

    def filter(self, graph: Data) -> Data:
        # assume the last 3 elements of edge attr are relative xyz normalized
        if graph.edge_attr.shape[-1] <= 3:
            logger.warning(f"{self.__class__} expects the last 3 elements of edge_attr to be relative xyz")
            return graph
        edge_rels = graph.edge_attr[:, :-3]
        relative_xyz = graph.edge_attr[:, -3:]
        keep_mask = edge_rels.any(dim=1)
        keep_mask |= (torch.linalg.norm(relative_xyz, ord=2, dim=1) ** 2) < self.distance_threshold

        graph.edge_index= graph.edge_index[:, keep_mask]
        graph.edge_attr = graph.edge_attr[keep_mask, :]
        return graph
