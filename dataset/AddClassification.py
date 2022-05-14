from dataclasses import dataclass, field
from dataset import object_classifications, ObjectClassification, Objects3DSSG
import logging
import torch
from torch_geometric.data import Data
import gin


logger = logging.getLogger(__name__)


@gin.configurable
@dataclass(frozen=True)
class AddClassification:
    convention: str
    rio_to_all: dict[int, ObjectClassification] = field(default_factory=lambda:{c.rio : c for c in object_classifications})
    num_classes: dict[str, int] = field(default_factory = lambda: {
        "eigen13": 14,  # 13 is window
        "rio27": 28,    # 27 is blanket
        "nyu40": 40,
        "rio": len(Objects3DSSG),
    })

    def one_hot_encode(self, rio_cls):
        _cl = rio_cls.type(torch.int64)  # 1 based
        _cl = _cl - 1
        if self.convention != "rio":
            _cl = torch.tensor([getattr(self.rio_to_all[int(c)],self.convention) for c in _cl], dtype=torch.long)
            if self.convention == "nyu40":
                _cl[_cl==40] = 0;  # remap to make proper one-hot
        embedding = torch.nn.functional.one_hot(_cl.flatten(), num_classes=self.num_classes[self.convention]).type(torch.float32)
        return embedding


    def __call__(self, graph: Data) -> Data:
        embedding = self.one_hot_encode(graph)
        graph.x = torch.hstack((graph.x, embedding))
        return graph

