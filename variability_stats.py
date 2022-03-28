import os
import json
from extract_data import build_graph_time_series
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

SAVE_TENSORS = False
MODE = "node"
data_folder = "data"


def parse_classes(node_file, edge_file):
    obj_labels = []
    edge_labels = []
    with open(node_file) as f:
        lines = f.readlines()
        for line in lines:
            obj_labels.append(line.split("\t")[1])

    with open(edge_file) as f:
        lines = f.readlines()
        for line in lines:
            edge_labels.append(line.split("\n")[0])

    return obj_labels, edge_labels


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
    return cov, len(pos_ts)-1


def calc_node_variability(nodes, node_label_dict, state_variability, location_covariance, location_counts, obj_labels):

    for node_id in nodes.keys():
        locations = nodes[node_id]["location"]
        states = nodes[node_id]["state"]
        label_idx = obj_labels.index(node_label_dict[node_id])
        if len(locations) > 1:
            cov_mat, n = calc_pos_variance(locations)
            location_counts[label_idx] += n
            alpha = n / location_counts[label_idx]
            location_covariance[label_idx, :, :] = alpha * cov_mat + (1-alpha) * location_covariance[label_idx, :, :]

        if len(states) > 1:
            changes, n = calc_att_variability(states)
            state_variability[label_idx, :] += [changes, n]

    return location_covariance, state_variability, location_counts


def calc_edge_variability():
    pass


if __name__ == "__main__":
    scans = json.load(open(os.path.join(data_folder, "3RScan.json")))
    objects_in_file = os.path.join(data_folder, "scene-graphs", "objects.json")
    relationships_in_file = os.path.join(data_folder, "scene-graphs", "relationships.json")
    graph_out_folder = os.path.join(data_folder, "graphs")
    node_class_file = os.path.join(data_folder, "scene-graphs", "classes.txt")
    edge_class_file = os.path.join(data_folder, "scene-graphs", "relationships.txt")
    obj_labels, edge_labels = parse_classes(node_class_file, edge_class_file)
    num_objects = len(obj_labels)
    num_relationships = len(edge_labels)
    state_variability = np.zeros((num_objects, 2))
    location_cov = np.zeros((num_objects, 3, 3))
    location_counts = np.zeros(num_objects)
    relationship_var = np.zeros((num_relationships, 2))

    for scene in tqdm(scans):
        graph_ts, node_ts, edge_ts, node_label_dict = build_graph_time_series(scene, objects_in_file, relationships_in_file)
        if len(node_ts.keys()) > 0 and (MODE == "node" or MODE == "both"):
            location_cov, state_variability, location_counts = calc_node_variability(node_ts, node_label_dict, state_variability, location_cov, location_counts, obj_labels)

        if len(edge_ts) > 0 and (MODE == "edge" or MODE == "both"):
            pass

    if SAVE_TENSORS:
        with open('data/results/state_variability.npy', 'wb') as f:
            np.save(f, state_variability)

        with open('data/results/location_counts.npy', 'wb') as f:
            np.save(f, location_counts)

        with open('data/results/location_cov.npy', 'wb') as f:
            np.save(f, location_cov)

