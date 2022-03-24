import os
import json
from extract_data import build_graph_time_series
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt


data_folder = "data"
visualize = False
obj_labels = []
att_labels = []
with open(os.path.join(data_folder, "scene-graphs", "classes.txt")) as f:
    lines = f.readlines()
    for line in lines:
        obj_labels.append(line.split("\t")[1])

num_objects = len(obj_labels)
num_atts = len(att_labels)


if __name__ == "__main__":
    with open('data/results/state_variability.npy', 'rb') as f:
        state_variability = np.load(f)

    with open('data/results/location_counts.npy', 'rb') as f:
        location_counts = np.load(f)

    with open('data/results/location_cov.npy', 'rb') as f:
        location_cov = np.load(f)

    state_var_ratio = state_variability[:, 0]/state_variability[:, 1]
    non_nan_idx = np.logical_not(np.isnan(state_var_ratio))
    state_var_ratio_no_nan = state_var_ratio[non_nan_idx]

    plt.figure()
    n, bins, patches = plt.hist(location_counts[location_counts < 100], 50, density=False, facecolor='g', alpha=0.75)
    plt.xlabel('# of Samples')
    plt.ylabel('Frequency per Object')
    plt.title('Histogram of Samples per Object')
    plt.savefig("data/results/sample_hist.jpg")
    plt.show()

    plt.figure()
    n, bins, patches = plt.hist(state_var_ratio_no_nan, 50, density=False, facecolor='b', alpha=0.75)
    plt.xlabel('State Variability Probability')
    plt.ylabel('Frequency per Object')
    plt.title('Histogram of State Variability per Object')
    plt.savefig("data/results/state_var_hist.png")
    plt.show()

    plt.figure()
    x_cov = location_cov[:, 0, 0]
    n, bins, patches = plt.hist(x_cov[x_cov < 50], 50, density=False, facecolor='r', alpha=0.75)
    plt.xlabel('X Variance')
    plt.ylabel('Frequency per Object')
    plt.title('Histogram of X Variance per Object')
    plt.savefig("data/results/x_var_hist.png")
    plt.show()

    plt.figure()
    y_cov = location_cov[:, 1, 1]
    n, bins, patches = plt.hist(y_cov[y_cov < 50], 50, density=False, facecolor='c', alpha=0.75)
    plt.xlabel('Y Variance')
    plt.ylabel('Frequency per Object')
    plt.title('Histogram of Y Variance per Object')
    plt.savefig("data/results/y_var_hist.png")
    plt.show()

    plt.figure()
    z_cov = location_cov[:, 2, 2]
    n, bins, patches = plt.hist(z_cov[z_cov < 1], 50, density=False, facecolor='m', alpha=0.75)
    plt.xlabel('Z Variance')
    plt.ylabel('Frequency per Object')
    plt.title('Histogram of Z Variance per Object')
    plt.savefig("data/results/z_var_hist.png")
    plt.show()

    i = 5
    state_thresh = state_variability[:, 1] > 10
    filtered_state_var = state_var_ratio[state_thresh]
    max_state_var = filtered_state_var.argsort()[-i:][::-1]
    non_nan_labels = np.asarray(obj_labels)[state_thresh]
    print('Top 5 Object State Variability:')
    for j in range(i):
        label = non_nan_labels[max_state_var[j]]
        print("{}: {}".format(j+1, label))
    print("\n")

    i = 5
    loc_total_var = np.linalg.norm(location_cov, axis=(1, 2))
    max_loc_var = loc_total_var.argsort()[-i:][::-1]
    print('Top 5 Object Location Variance:')
    for j in range(i):
        label = obj_labels[max_loc_var[j]]
        print("{}: {}".format(j, label))
    print("\n")

