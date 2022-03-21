import os
import json
import networkx as nx
from data_vis import load_scene_mesh, visualize_graph

data_folder = "data"
visualize = False


def parse_node(in_node, node_dict):
    for attribute in in_node.keys():
        if attribute in node_dict.keys():
            node_dict[attribute].append(in_node[attribute])
        else:
            node_dict[attribute] = [in_node[attribute]]
    return node_dict


def parse_nodelist(in_nodes, nodes_dict):
    for node in in_nodes:
        id = node[0]
        content = node[1]["attributes"]
        if id in nodes_dict.keys():
            nodes_dict[id] = parse_node(content, nodes_dict[id])
        else:
            nodes_dict[id] = parse_node(content, {})

    return nodes_dict


def format_scan_dict(unformated_dict, attribute):
    scan_list = unformated_dict["scans"]
    formatted_dict = {}
    for scan in scan_list:
        formatted_dict[scan["scan"]] = scan[attribute]
    return formatted_dict


def format_sem_seg_dict(sem_seg_dict):
    object_dict = {}
    for object in sem_seg_dict["segGroups"]:
        object_dict[object["id"]] = object["obb"]["centroid"]

    return object_dict


def build_scene_graph(nodes_dict, edges_dict, scan_id):
    if scan_id not in nodes_dict.keys() or scan_id not in edges_dict.keys():
        print("No graph information for this scan")
        return None, None, None

    scan_sem_seg_file = os.path.join(data_folder, "scans", scan_id, "semseg.v2.json")
    if os.path.isfile(scan_sem_seg_file):
        semantic_seg = json.load(open(scan_sem_seg_file))
        object_pos_list = format_sem_seg_dict(semantic_seg)
    else:
        print("No Semantic Segmentation File Available")
        object_pos_list = None

    graph = nx.Graph()
    nodes = nodes_dict[scan_id]
    input_node_list = []
    label_dict = {}
    for node in nodes:
        id = int(node["id"])
        att_dict = {"label": node.pop("label", None), "affordances": node.pop("affordances", None),
                                  "attributes": node.pop("attributes", None), "global_id": node.pop("global_id", None),
                                  "color": node.pop("ply_color", None)}

        if object_pos_list is not None:
            att_dict["location"]: object_pos_list[id]
        input_node_list.append((id, att_dict))
        label_dict[node["id"]] = att_dict["label"]
    graph.add_nodes_from(input_node_list)
    edges = edges_dict[scan_id]
    for edge in edges:
        graph.add_edge(edge[0], edge[1])

    return graph, input_node_list, edges


if __name__ == "__main__":
    scans = json.load(open(os.path.join(data_folder, "3RScan.json")))
    objects = json.load(open(os.path.join(data_folder, "scene-graphs", "objects.json")))
    objects_dict = format_scan_dict(objects, "objects")
    relationships = json.load(open(os.path.join(data_folder, "scene-graphs", "relationships.json")))
    relationships_dict = format_scan_dict(relationships, "relationships")

    graph_out_folder = os.path.join(data_folder, "graphs")

    for scene in scans:
        ref_scan = scene["reference"]
        follow_scans = scene["scans"]
        scene_graph, nodes, edges = build_scene_graph(objects_dict, relationships_dict, ref_scan)
        scene_graphs_over_time = [scene_graph]
        nodes_over_time = {}
        nodes_over_time = parse_nodelist(nodes, nodes_over_time) if nodes is not None else None
        edges_over_time = [edges]
        if visualize:
            load_scene_mesh(ref_scan)
            visualize_graph(scene_graph, graph_out_folder, ref_scan)

        for scan in follow_scans:
            scan_exists = os.path.isdir(os.path.join(data_folder, "scans", scan["reference"]))
            scan_id = scan["reference"]
            scene_graph, nodes, edges = build_scene_graph(objects_dict, relationships_dict, scan_id)
            scene_graphs_over_time.append(scene_graph)
            nodes_over_time = parse_nodelist(nodes, nodes_over_time) if nodes is not None else None
            edges_over_time.append(edges)

            if visualize:
                load_scene_mesh(ref_scan)
                visualize_graph(scene_graph, graph_out_folder, ref_scan)


