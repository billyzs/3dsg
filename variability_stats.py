import os
import json
from extract_data import build_graph_time_series
import numpy as np
from tqdm import tqdm

data_folder = "data"
visualize = False
obj_labels = []
att_labels = []
with open(os.path.join(data_folder, "scene-graphs", "classes.txt")) as f:
    lines = f.readlines()
    for line in lines:
        obj_labels.append(line.split("\t")[1])

with open(os.path.join(data_folder, "scene-graphs", "attributes.txt")) as f:
    lines = f.readlines()
    for line in lines:
        att_labels.append(line.split(":")[0])
    att_labels = list(set(att_labels))

num_objects = len(obj_labels)
num_atts = len(att_labels)


def calc_att_variability(att_ts):
    total_transitions = len(att_ts)-1
    changes = 0
    for i in range(total_transitions):
        if att_ts[i] != att_ts[i+1]:
            changes += 1

    return changes, total_transitions


def calc_pos_variance(pos_ts):
    np_pos = np.asarray(pos_ts).T
    cov = np.cov(np_pos)
    return cov, len(pos_ts)


def calc_node_variability(nodes):
    variability_dict = {}
    for node_id in nodes.keys():
        node_atts = nodes[node_id]
        node_var_dict = {}
        ts_size = len(list(node_atts.values())[0])
        if ts_size > 1:
            for att in node_atts.keys():
                if att == "location":
                    node_var_dict[att] = calc_pos_variance(node_atts[att])
                else:
                    node_var_dict[att] = calc_att_variability(node_atts[att])

        variability_dict[node_id] = node_var_dict

    return variability_dict


def parse_variability(variability_matrix, new_var_stats, node_label_dict):
    for node_id in new_var_stats.keys():
        var_stats = new_var_stats[node_id]
        label_idx = obj_labels.index(node_label_dict[node_id])
        for att in var_stats.keys():
            if att != "location":
                att_idx = att_labels.index(att)
                att_stats = var_stats[att]
                variability_matrix[label_idx, att_idx, :] += att_stats

    return variability_matrix


if __name__ == "__main__":
    scans = json.load(open(os.path.join(data_folder, "3RScan.json")))
    objects_in_file = os.path.join(data_folder, "scene-graphs", "objects.json")
    relationships_in_file = os.path.join(data_folder, "scene-graphs", "relationships.json")
    graph_out_folder = os.path.join(data_folder, "graphs")
    variability_matrix = np.zeros((num_objects, num_atts, 2))

    for scene in tqdm(scans):
        graph_ts, node_ts, edge_ts, node_label_dict = build_graph_time_series(scene, objects_in_file, relationships_in_file)
        if len(node_ts.keys()) > 0:
            new_var_stats = calc_node_variability(node_ts)
            variability_matrix = parse_variability(variability_matrix, new_var_stats, node_label_dict)

    print("done")


