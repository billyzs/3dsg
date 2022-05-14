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
    given an absolute distance threshold, remove edges that both
    (do not have any relations
    and have relative distance smaller than this threshold)
    this should be done before any attribute selection
    """
    abs_distance_threshold: float
    enabled: bool = False
    normalize: bool = True
    remove_relative_distance: bool = False
    def __call__(self, graph: Data) -> Data:
        if self.enabled:
            graph = self.filter(graph)
        if self.normalize:
            graph = self._normalize(graph)
        if self.remove_relative_distance:
            graph = self._remove_relative_distance(graph)
        return graph

    def _remove_relative_distance(self, graph: Data) -> Data:
       graph.edge_attr = graph.edge_attr[:, :-3]
       return graph

    def _normalize(self, graph: Data) -> Data:
        # should be done after filtering
        xyz = graph.edge_attr[:, -3:]
        max_dist = torch.max(xyz)
        graph.edge_attr[:, -3:] = xyz / max_dist
        return graph

    def filter(self, graph: Data) -> Data:
        # assume the last 3 elements of edge attr are relative xyz
        if graph.edge_attr.shape[-1] <= 3:
            logger.warning(f"{self.__class__} expects the last 3 elements of edge_attr to be relative xyz")
            return graph
        edge_rels = graph.edge_attr[:, :-3]
        xyz = graph.edge_attr[:, -3:]
        keep_mask = edge_rels.any(dim=1)  # keep any edges w/ existing relationships
        keep_mask |= (torch.linalg.norm(xyz, ord=2, dim=1)) < self.abs_distance_threshold

        graph.edge_index= graph.edge_index[:, keep_mask]
        graph.edge_attr = graph.edge_attr[keep_mask, :]
        return graph
