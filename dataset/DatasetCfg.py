from dataclasses import dataclass
from typing import List

# Contains all parameters for scene variability estimation dataset
@dataclass
class DatasetCfg:
    # Directory containing all raw data as per the specified file structure
    root: str = "/cluster/work/riner/users/PLR-2022/3dssg/3RScan"

    # Parameters for attribute embedding
    class AttributeParams:
        allowed_attributes: List[str] = [
            "color:white",
            "color:black",
            "color:green",
            "color:blue",
            "color:red",
            "color:brown",
            "color:yellow",
            "color:gray",
            "color:orange",
            "color:purple",
            "color:pink",
            "color:beige",
            "color:bright",
            "color:dark",
            "color:light",
            "color:silver",
            "color:gold",
            "shape:round",
            "shape:flat",
            "shape:L-shaped",
            "shape:semicircular",
            "shape:circular",
            "shape:square",
            "shape:rectangular",
            "shape:sloping",
            "shape:cylindrical",
            "shape:oval",
            "shape:bunk",
            "shape:heart-shaped",
            "shape:u-shaped",
            "shape:octagon",
            "style:classy",
            "style:classical",
            "style:minimalistic",
            "state:new",
            "state:old",
            "state:dirty",
            "state:clean",
            "state:open",
            "state:empty",
            "state:full",
            "state:hanging",
            "state:half full/empty",
            "state:closed",
            "state:half open/closed",
            "state:messy",
            "state:tidy",
            "state:on",
            "state:off",
            "state:folded together",
            "state:seat up",
            "state:seat down",
            "state:up",
            "state:down",
            "state:half up/down",
            "state:bare",
            "state:written on",
            "size:small",
            "size:big",
            "size:tall",
            "size:low",
            "size:narrow",
            "size:wide",
            "material:wooden",
            "material:plastic",
            "material:metal",
            "material:glass",
            "material:stone",
            "material:leather",
            "material:concrete",
            "material:ceramic",
            "material:brick",
            "material:padded",
            "material:cardboard",
            "material:marbled",
            "material:carpet",
            "material:cork",
            "material:velvet",
            "texture:striped",
            "texture:patterned",
            "texture:dotted",
            "texture:colorful",
            "texture:checker",
            "texture:painted",
            "texture:shiny",
            "texture:tiled",
            "other:mobile",
            "other:rigid",
            "other:nonrigid",
            "symmetry:no symmetry",
            "symmetry:1 plane",
            "symmetry:2 planes",
            "symmetry:infinite planes"
        ]
        global_loc: bool = False

    attributes: AttributeParams = AttributeParams()

    # Parameters for object embedding
    class ObjectParams:
        mode: str = "full"

    objects: ObjectParams = ObjectParams()

    # Parameters for edge embeddings
    class RelationshipParams:
        allowed_relationships: List[str] = [
            "none",
            "supported by",
            "left",
            "right",
            "front",
            "behind",
            "close by",
            "inside",
            "bigger than",
            "smaller than",
            "higher than",
            "lower than",
            "same symmetry as",
            "same as",
            "attached to",
            "standing on",
            "lying on",
            "hanging on",
            "connected to",
            "leaning against",
            "part of",
            'belonging to',
            "build in",
            "standing in",
            "cover",
            "lying in",
            "hanging in",
            "same color",
            "same material",
            "same texture",
            "same shape",
            "same state",
            "same object type",
            "messier than",
            "cleaner than",
            "fuller than",
            "more closed",
            "more open",
            "brighter than",
            "darker than",
            "more comfortable than"
        ]
        relative_loc: bool = True

    relationships: RelationshipParams = RelationshipParams()

    # Parameters for variability embedding
    class VariabilityParams:
        threshold: float = 0.1

    variability: VariabilityParams = VariabilityParams()
