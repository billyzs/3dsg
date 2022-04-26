from typing import List
from collections.abc import Sequence
from dataclasses import dataclass, field, InitVar
from torch_geometric.data import Data
from Relationships3DSSG import Relationships3DSSG
from Attributes3DSSG import Attributes3DSSG
from enum import IntEnum
import torch
import logging
import gin


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AllowList:
    allowed_items: List[str]
    allow_mask: Sequence[bool] = field(init=False)
    pre_offset: int
    post_offset: int


    def __post_init__(self):
        s = set([self.enum_cls.key_to_value(r) for r in self.allowed_items])
        mask = self.enum_cls.binary_encode(self.allowed_items)
        super().__setattr__("allow_mask", \
            [True] * self.pre_offset +\
            mask +\
            [True] * self.post_offset)


@gin.configurable
class RelationshipsAllowList(AllowList):
    enum_cls: IntEnum = Relationships3DSSG
    def __call__(self, graph: Data) -> Data:
        graph.edge_attr = graph.edge_attr[:, self.allow_mask]
        return graph


@gin.configurable
class AttributesAllowList(AllowList):
    enum_cls: IntEnum = Attributes3DSSG
    def __call__(self, graph: Data) -> Data:
        graph.x= graph.x[:, self.allow_mask]
        return graph


@gin.configurable
@dataclass
class TransformPipeline:
    steps: InitVar([])
    def __post_init__(self, steps):
        self.instances = [c() for c in steps]


    def __call__(self, arg):
        for proc in self.instances:
            arg = proc(arg)
        return arg


if __name__ == "__main__":
    import copy
    from SceneGraphChangeDataset import SceneGraphChangeDataset
    config = [
        "SceneGraphChangeDataset.root = '/home/bzs/devel/euler/3dssg/3RScan/'",
        "AttributesAllowList.allowed_items = ['state_open', 'state_closed']",
        "AttributesAllowList.pre_offset = 0",
        "AttributesAllowList.post_offset = 0",
        "RelationshipsAllowList.allowed_items = ['supported_by', 'connected_to']",
        "RelationshipsAllowList.pre_offset = 0",
        "RelationshipsAllowList.post_offset = 3",
        "TransformPipeline.steps = [@RelationshipsAllowList, @AttributesAllowList]",
    ]
    gin.parse_config(config)
    dataset = SceneGraphChangeDataset()
    ra = TransformPipeline()

    # import pickle
    # with open("complete_graph.pickle", 'rb') as f:
        # g = pickle.load(f)
    # gp = copy.deepcopy(g)
    # gp = ra(g)

