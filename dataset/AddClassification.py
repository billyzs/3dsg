from dataclasses import dataclass
import logging
import torch
from torch_geometric.data import Data
import gin


logger = logging.getLogger(__name__)


@dataclass
@gin.configurable
class AddClassification:
    method: str = "simple"
    def _simple_add_classification(self, graph: Data) -> Data:
        graph.x = torch.hstack((graph.x, graph.classifications))
        return graph


    def __call__(self, graph: Data) -> Data:
        if self.method == "simple":
            graph = self._simple_add_classification(graph)
        return graph

