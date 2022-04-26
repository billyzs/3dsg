from dataclasses import dataclass
from typing import List

# Contains all parameters for scene variability estimation dataset
@dataclass
class DatasetCfg:
    # Directory containing all raw data as per the specified file structure
    root: str = "/cluster/work/riner/users/PLR-2022/3dssg/3RScan"

    # Parameters for attribute embedding
    class AttributeParams:
        global_loc: bool = False

    attributes: AttributeParams = AttributeParams()

    # Parameters for object embedding
    class ObjectParams:
        mode: str = "full"

    objects: ObjectParams = ObjectParams()

    # Parameters for edge embeddings
    class RelationshipParams:
        relative_loc: bool = True

    relationships: RelationshipParams = RelationshipParams()

    # Parameters for variability embedding
    class VariabilityParams:
        threshold: float = 0.1

    variability: VariabilityParams = VariabilityParams()
