from collections import OrderedDict
from typing import List, Dict, Optional
from torch_geometric.data import Data
from Attributes3DSSG import Attributes3DSSG
import gin
import logging


logger = logging.getLogger(__name__)


class AttributesAllowList:
    def __init__(
        self,
        allowed_attributes: Optional[List[str]] = None,
    ):
        self.allowlist = allowed_attributes
        self.attributes = self._validate_input(allowed_attributes)

    def _validate_input(self, allowed_attributes: Optional[List[str]]):
        if allowed_attributes is None:
            raise ValueError("no attributes specified to build node embedding")
        logger.info(f"processing {allowed_attributes=}")
        attributes = OrderedDict()  # import to keep order for consistency

        # will raise if input invalid
        attr_enum = {Attributes3DSSG.to_enum(a): a for a in allowed_attributes}
        enums = sorted(list(attr_enum.keys()))
        for e in enums:
            attr = attr_enum[e]
            category, val = attr.split(":")
            attributes.setdefault(category, []).append(val)
        logger.info(f"allowing the following attributes: {attributes}")
        return attributes

    def __call__(self, graph: Data) -> Data:
        return graph

