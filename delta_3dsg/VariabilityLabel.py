from dataclasses import dataclass
from torch_geometric.data import Data
from Relationships3DSSG import Relationships3DSSG
import torch
import logging
import gin


logger = logging.getLogger(__name__)

# makes __init__ params configurable; see variability_example.gin
@gin.configurable
@dataclass  # implicit __init__
class VariabilityLabel:
    moved_distance_threshold: float

    def make_mask(self, graph) -> torch.Tensor:
        # TODO sam implement logic here
        pass

    def make_variability_label(self, graph: Data) -> torch.Tensor:
        scan_id = graph.input_graph
        output_graph_id = graph.output_graph
        num_nodes = graph.x.shape()[0]
        label = torch.zeros((num_nodes, ))
        # TODO sam implement logic here
        return label


    def __call__(self, graph: Data) -> Data:
        graph.y = self.make_variabilit_label(graph)
        graph.mask = self.make_mask(graph)
        graph.moved_distance_threshold = self.moved_distance_threshold
        return graph


if __name__ == "__main__":
    gin.parse_config_file("variability_example.gin")
    vl = VariabilityLabel()  # just leave blank; let config file choose value
    print(vl.moved_distance_threshold)

