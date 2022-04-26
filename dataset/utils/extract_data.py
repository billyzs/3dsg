import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
import torch


def visualize_graph(*a, **kw):
    try:
        from dataset.utils.data_vis import visualize_graph as v
        v(*a, **kw)
    except ImportError:
        pass

def transform_locations(node_list: List[Tuple], T: torch.Tensor) -> List[Tuple]:
    for i in range(len(node_list)):
        loc = node_list[i][1]["attributes"]["location"]
        homogenous_location = torch.reshape(torch.Tensor([loc[0], loc[1], loc[2], 1]), (4, 1))
        # trans_mat = np.asarray(T).reshape((4, 4)).T
        ref_frame_location = T @ homogenous_location
        node_list[i][1]["attributes"]["location"] = ref_frame_location[:3]

    return node_list


def parse_nodelist(in_nodes, nodes_dict, label_dict):
    for node in in_nodes:
        id = node[0]
        if id not in nodes_dict.keys():
            nodes_dict[id] = {"location": [], "state": []}
            label_dict[id] = node[1]["label"]
        if "location" in node[1]["attributes"].keys():
            location = node[1]["attributes"]["location"]
            nodes_dict[id]["location"].append(location)

        if "state" in node[1]["attributes"].keys():
            state = node[1]["attributes"]["state"]
            nodes_dict[id]["state"].append(state)

    return nodes_dict, label_dict


def format_scan_dict(unformated_dict: Dict, attribute: str) -> Dict:
    scan_list = unformated_dict["scans"]
    formatted_dict = {}
    for scan in scan_list:
        formatted_dict[scan["scan"]] = scan[attribute]
    return formatted_dict


def format_sem_seg_dict(sem_seg_dict: Dict) -> Dict:
    object_dict = {}
    for object in sem_seg_dict["segGroups"]:
        object_dict[object["id"]] = object["obb"]["centroid"]

    return object_dict


def build_scene_graph(nodes_dict: Dict, edges_dict: Dict, scan_id: str, root: str, graph_out=False) -> (Optional, List[Tuple], List[Tuple]):
    if scan_id not in nodes_dict.keys() or scan_id not in edges_dict.keys():
        # print("No graph information for this scan")
        return None, None, None

    scan_sem_seg_file = os.path.join(root, "raw", "semantic_segmentation_data", scan_id, "semseg.v2.json")
    if os.path.isfile(scan_sem_seg_file):
        semantic_seg = json.load(open(scan_sem_seg_file))
        object_pos_list = format_sem_seg_dict(semantic_seg)
    else:
        print(f"No Semantic Segmentation File Available for {scan_id}")
        return None, None, None


    nodes = nodes_dict[scan_id]
    input_node_list = []
    for node in nodes:
        node_copy = node.copy()
        id = int(node["id"])
        att_dict = {"label": node_copy.pop("label", None), "affordances": node_copy.pop("affordances", None),
                                  "attributes": node_copy.pop("attributes", None), "global_id": node_copy.pop("global_id", None),
                                  "color": node_copy.pop("ply_color", None)}

        if object_pos_list is not None:
            att_dict["attributes"]["location"] = np.clip(object_pos_list[id], -100, 100)

        att_dict["attributes"].pop("lexical", None)
        input_node_list.append((id, att_dict))
    edges = edges_dict[scan_id]
    if graph_out:
        graph = nx.Graph()
        graph.add_nodes_from(input_node_list)
        for edge in edges:
            graph.add_edge(edge[0], edge[1])
    else:
        graph = None

    return graph, input_node_list, edges


def build_graph_time_series(scene_dict, objects_file, relationships_file, visualize=False):
    objects_json = json.load(open(objects_file))
    relationships_json = json.load(open(relationships_file))
    objects_dict = format_scan_dict(objects_json, "objects")
    relationships_dict = format_scan_dict(relationships_json, "relationships")

    ref_scan = scene_dict["reference"]
    follow_scans = scene_dict["scans"]
    scene_graph, nodes, edges = build_scene_graph(objects_dict, relationships_dict, ref_scan, "")
    scene_graphs_over_time = [scene_graph]
    if nodes is not None:
        nodes_over_time, node_label_dict = parse_nodelist(nodes, {}, {})
    else:
        nodes_over_time, node_label_dict = {}, {}

    edges_over_time = [edges]
    if visualize:
        visualize_graph(scene_graph, graph_out_folder, ref_scan)

    for scan in follow_scans:
        scan_id = scan["reference"]
        scene_graph, nodes, edges = build_scene_graph(objects_dict, relationships_dict, scan_id, "")
        scene_graphs_over_time.append(scene_graph)
        if nodes is not None:
            transformed_nodes = transform_locations(nodes, scan["transform"])
            nodes_over_time, node_label_dict = parse_nodelist(transformed_nodes, nodes_over_time, node_label_dict)
        edges_over_time.append(edges)
        if visualize:
            visualize_graph(scene_graph, graph_out_folder, ref_scan)

    return scene_graphs_over_time, nodes_over_time, edges_over_time, node_label_dict


if __name__ == "__main__":
    data_folder = "/home/sam/ethz/plr/plr-2022-predicting-changes/data/raw"

    scans = json.load(open(os.path.join(data_folder, "3RScan.json")))
    objects_in_file = os.path.join(data_folder, "scene-graphs", "objects.json")
    relationships_in_file = os.path.join(data_folder, "scene-graphs", "relationships.json")

    graph_out_folder = os.path.join(data_folder, "graphs")

    for scene in scans:
        scenegraph_ts, node_ts, edge_ts, label_dict = build_graph_time_series(scene, objects_in_file, relationships_in_file)
