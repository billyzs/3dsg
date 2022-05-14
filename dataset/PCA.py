from dataclasses import dataclass
import logging
import gin
import torch

logger = logging.getLogger(__name__)


@gin.configurable
@dataclass
class PCATransform:
    def __call__(self, graph):
        graph.node_attributes = graph.x
        graph.x = graph.pca.type(torch.float)
        return graph
