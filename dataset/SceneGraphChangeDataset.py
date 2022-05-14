import torch
import os
from torch_geometric.data import InMemoryDataset, Data
import json
from .AttributeEmbedding import BinaryNodeEmbedding
from .RelationshipEmbedding import BinaryEdgeEmbedding
from .VariabilityEmbedding import BinaryVariabilityEmbedding
from .DatasetCfg import DatasetCfg
from .utils.extract_data import build_scene_graph, format_scan_dict, transform_locations
from typing import List, Dict
import gin


def get_scene_list(scene: Dict) -> (List[str], List[torch.Tensor]):
    scan_id_set = [scene["reference"]]
    scan_tf_set = [torch.eye(4)]
    for follow_scan in scene["scans"]:
        scan_id_set.append(follow_scan["reference"])
        if "transform" in follow_scan.keys():
            scan_tf_set.append(torch.Tensor(follow_scan["transform"]).reshape((4, 4)).T)
        else:
            scan_tf_set.append(torch.eye(4))

    return scan_id_set, scan_tf_set


@gin.configurable
class SceneGraphChangeDataset(InMemoryDataset):
    def __init__(self, root=None, cfg: DatasetCfg = None, transform=None, pre_transform=None, pre_filter=None):
        self.cfg: DatasetCfg = cfg
        if not root:
            root = self.cfg.root
        if root and cfg:
            self.cfg.root = root
        self.root = root
        self.raw_files: str = os.path.join(root, "raw", "raw_files.txt")
        super().__init__(self.root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def _load_raw_files(self):
        # Load raw data files as per standard dataset folder organization
        if self.cfg:
            root: str = self.cfg.root
        else:
            root = self.root
        self.scans: List[Dict] = json.load(open(os.path.join(root, "raw", "3RScan.json")))
        object_data: Dict = json.load(open(os.path.join(root, "raw", "scene-graphs", "objects.json")))
        relationship_data: Dict = json.load(open(os.path.join(root, "raw", "scene-graphs", "relationships.json")))
        self.objects_dict: Dict[List[Dict]] = format_scan_dict(object_data, "objects")
        self.relationships_dict: Dict[List[Dict]] = format_scan_dict(relationship_data, "relationships")

        self.node_embedder: BinaryNodeEmbedding = BinaryNodeEmbedding(att_cfg=DatasetCfg.attributes, obj_cfg=DatasetCfg.objects)
        self.edge_embedder: BinaryEdgeEmbedding = BinaryEdgeEmbedding(cfg=DatasetCfg.relationships)
        self.variability_embedder: BinaryVariabilityEmbedding = BinaryVariabilityEmbedding(cfg=DatasetCfg.variability)

    @property
    def raw_file_names(self):
        if os.path.isfile(self.raw_files):
            with open(self.raw_files) as f:
                files = f.read().splitlines()
        else:
            files = ["3RScan.json",
                     "scene-graphs/objects.json",
                     "scene-graphs/relationships.json",
                     ]
        return files

    @property
    def processed_file_names(self):
        return ['scene_graph_data.pt']

    def download(self):
        raise Exception("Files Not Found. Download dataset files as per standard format")

    def process(self):
        self._load_raw_files()
        samples = []
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = lambda x: (_ for _ in x)
        for scene in tqdm(self.scans):
            scan_id_set, scan_tf_set = get_scene_list(scene)
            for i in range(len(scan_id_set)):
                for j in range(len(scan_id_set)):
                    if i != j:
                        _, nodes_1, edges_1 = build_scene_graph(self.objects_dict, self.relationships_dict,
                                                                scan_id_set[i], self.root)
                        _, nodes_2, edges_2 = build_scene_graph(self.objects_dict, self.relationships_dict,
                                                                scan_id_set[j], self.root)
                        T_1I = scan_tf_set[i]
                        T_2I = scan_tf_set[j]
                        if nodes_1 is not None and nodes_2 is not None:
                            transf_node_1 = transform_locations(nodes_1, T_1I)
                            transf_node_2 = transform_locations(nodes_2, T_2I)
                            node_embeddings, node_idxs, node_pos, node_classifications = self.node_embedder.generate_node_embeddings(
                                transf_node_1)
                            edge_embeddings, edge_idxs = self.edge_embedder.generate_edge_embeddings(nodes_1, edges_1,
                                                                                                     node_idxs)

                            node_labels, state_mask = self.variability_embedder.generate_variability_embedding(
                                transf_node_1, transf_node_2, node_idxs)

                            sample = Data(
                                x=node_embeddings,
                                edge_index=edge_idxs,
                                edge_attr=edge_embeddings,
                                y=node_labels,
                                pos=node_pos,
                                classifications=node_classifications,
                                input_graph=scan_id_set[i],
                                output_graph=scan_id_set[j],
                                input_tf=T_1I,
                                output_tf=T_2I,
                                state_mask=state_mask
                            )
                            samples.append(sample)

        data, slices = self.collate(samples)
        torch.save((data, slices), self.processed_paths[0])

