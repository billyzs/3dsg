#!/usr/bin/env python3

import os
import json
import itertools
import numpy as np
import meshio
import plotly.graph_objects as gobj
from dataset.Taxonomy3DSSG import Objects3DSSG

scan_id = "787ed580-9d98-2c97-8167-6d3b445da2c0"
scan_dir = "/home/bzs/devel/euler/3dssg/3RScan/"
ply_file = "labels.instances.annotated.v2.ply"

mesh_path = os.path.join(scan_dir, scan_id, ply_file)
mesh_data = meshio.read(mesh_path)


def mesh_to_figure(mesh_data: meshio.Mesh):
    vertices = mesh_data.points  # n by 3
    triangles = mesh_data.get_cells_type("triangle")
    x, y, z = vertices.T
    I, J, K = triangles.T
    rgb = np.array([mesh_data.point_data[c] for c in ['red', 'green', 'blue']])
    pl_mesh = gobj.Mesh3d(
        x=x, y=y, z=z,
        i=I, j=J, k=K,
        vertexcolor=rgb.T,
    )
    return pl_mesh


def scene_graph_to_figure():
    with open(os.path.join(scan_dir, "scene-graphs", "objects.json")) as f:
        all_scans = json.load(f)
    id_to_nodes = {s["scan"]: s["objects"] for s in all_scans["scans"]}
    if not id_to_nodes[scan_id]:
        print(f"nodes for {scan_id} not found!")
        return
    # get the sem seg obj centroids
    nodes_in_this_scan = id_to_nodes[scan_id]
    with open(os.path.join(scan_dir, scan_id, "semseg.v2.json")) as segf:
        seg = json.load(segf)
    objects = seg['segGroups']
    id_to_global_id = {int(i['id']): int(i["global_id"]) for i in nodes_in_this_scan}
    id_to_centroid = dict()
    for obj in objects:
        obj_id = int(obj["objectId"])
        obj_centroid = np.array(obj["obb"]["centroid"])
        id_to_centroid[obj_id] = obj_centroid
    ids_in_scan = list(id_to_centroid.keys())
    nodes = np.array(list(id_to_centroid.values())).transpose()
    hovertext = []
    edges, distances = [], []
    # distance graph is fully connected
    for (fr, to) in itertools.combinations(ids_in_scan, 2):  # make edges pairwise 
        fp = id_to_centroid[fr]
        tp = id_to_centroid[to]
        v = tp - fp
        dist = np.sqrt(np.dot(v, v))
        distances.append(dist)
        edges.extend([fp, tp])
        txt = f"{fr}->{to}:{dist:.2f}m"
        hovertext.extend([txt, txt])

    ex, ey, ez = np.array(edges).transpose()
    edges_trace = gobj.Scatter3d(
        x=ex, y=ey, z=ez,
        mode='lines',
        text=hovertext,
    )

    x, y, z = nodes
    nodes_trace = gobj.Scatter3d(
        x=x, y=y, z=z,
        text=[str(Objects3DSSG(id_to_global_id[i])) for i in ids_in_scan],
        mode='markers',
        marker=dict(
            showscale=True,
            color='#000',
        ),
    )
    return nodes_trace, edges_trace


def main():
    pl_mesh = mesh_to_figure(mesh_data)
    pl_mesh.name = f"{scan_id=}"
    pl_nodes, pl_edges = scene_graph_to_figure()
    fig = gobj.Figure(
        data=[
            pl_nodes, pl_edges,
            pl_mesh,
        ])
    fig.show()


if __name__ == "__main__":
    main()
