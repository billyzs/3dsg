#!/usr/bin/env python3

import os
import json
import itertools
import numpy as np
import meshio
import plotly.graph_objects as gobj
from dataset import *
from dataset.DistanceBasedPartialConnectivity import _DistanceBasedPartialConnectivity
from torch_geometric.data import Data
from collections.abc import Sequence
from models.SimpleGCN import SimpleMPGNN
import torch
import copy

ply_file = "labels.instances.annotated.v2.ply"
nl =  '<br>'

mesh_opacity = 0.5
edges_opacity = 0.3
root = ""
curr_graph = None
graph_stats_str: str = ""
ref_scan_to_rescan: dict[str, list[str]] = dict()
curr_refscan_id = "c92fb576-f771-2064-845a-a52a44a9539f"  # 11 rescans
curr_rescans = []
dist_filter = _DistanceBasedPartialConnectivity(enabled=True, normalize=False, abs_distance_threshold=1e9)
GCN = None
transform = None
connectivity_to_dist = {
    1: 1.680349,
    2: 2.625959,
    3: 3.730257,
    4: 1e9,
}

def summarize_graph(graph):
    s = str(graph)
    return s

def mesh_to_figure(mesh_data: meshio.Mesh):
    vertices = mesh_data.points.T # n by 3
    triangles = mesh_data.get_cells_type("triangle")
    try:
        x,y,z = vertices.T
    except ValueError:
        x,y,z = vertices
    I,J,K = triangles.T
    rgb = np.array([mesh_data.point_data[c] for c in ['red', 'green', 'blue']])
    pl_mesh = gobj.Mesh3d(
        x=x,y=y,z=z,
        i=I,j=J,k=K,
        vertexcolor=rgb.T,
        opacity=mesh_opacity,
        name="mesh",
    )
    return pl_mesh


def node_attr_to_str(node_attr: Sequence) -> str:
    # omit attributes that are 0
    attr_strs = nl.join([": ".join((str(Attributes3DSSG(idx)), str(int(val)))) for idx, val in enumerate(node_attr) if val])
    return attr_strs


def edge_rel_to_str(edge_vec: Sequence) -> str:
    # omit relation that is 0
    attr_strs = nl.join([": ".join((str(Relationships3DSSG(idx)), str(int(val)))) for idx, val in enumerate(edge_vec) if val])
    return attr_strs


def plot_mesh(root: str, graph: Data) -> gobj.Scatter3d:
    scan_id = graph.input_graph
    mesh_path = os.path.join(root, scan_id, ply_file)
    mesh_trace = None
    if os.path.isfile(mesh_path):
        mesh_data = meshio.read(mesh_path)
        mesh_trace = mesh_to_figure(mesh_data)
    return mesh_trace

def load_centroids(root: str, graph: Data) -> Data:
    with open(os.path.join(root, graph.input_graph, "semseg.v2.json")) as segf:
        seg = json.load(segf)
    objects = seg['segGroups']
    centroids = torch.zeros((graph.x.shape[0],3))
    # id_to_global_id = {int(i['id']): int(i["global_id"]) for i in nodes_in_this_scan}
    id_to_centroid = dict()
    for obj in objects:
        obj_id = int(obj["objectId"])
        obj_centroid = torch.tensor(obj["obb"]["centroid"]).flatten()
        id_to_centroid[obj_id] = obj_centroid
    ids_in_scan = sorted(list(id_to_centroid.keys()))
    for idx, obj_id in enumerate(ids_in_scan):
        centroids[idx, :] = id_to_centroid[obj_id]
    graph.pos = centroids
    return graph


def plot_edges(graph: Data) -> gobj.Scatter3d:
    total_rels = len(Relationships3DSSG)
    num_edges = graph.edge_index.shape[-1]
    edge_xyz = np.zeros((3, num_edges * 2))
    edge_hovertext = []
    axis = ["x", 'y', 'z']
    for idx, (fr, to) in enumerate(graph.edge_index.transpose(1,0)):
        fr, to = int(fr), int(to)
        edge_xyz[:, idx*2] = graph.pos[fr, :].flatten()
        edge_xyz[:, idx*2+1] = graph.pos[to, :].flatten()
        edge_rel = graph.edge_attr[idx, :-3]  # assume last 3 are relative xyz
        fr_obj = str(Objects3DSSG(int(graph.classifications[fr])))
        to_obj = str(Objects3DSSG(int(graph.classifications[to])))
        text = f"{fr}[{fr_obj}]->{to}[{to_obj}]{nl}"
        if len(edge_rel) > total_rels:
            edge_rel, relative_xyz = edge_rel[:total_rels], edge_rel[total_rels:]
            text = text + nl.join((f"{axis[idx]}: {relative_xyz[idx]}" for idx in range(3)))
        edge_attr_txt = edge_rel_to_str(edge_rel)
        text = nl.join((text, edge_attr_txt))
        edge_hovertext.append(text)
    ex, ey, ez = edge_xyz
    edge_dist = np.linalg.norm(edge_xyz, axis=0, ord=2).flatten()
    edge_hovertext= [t + f"{nl}{edge_dist[idx]:.3f} m" for idx, t in enumerate(edge_hovertext)]
    edge_colors = edge_dist - (edge_dist.max() - 0.01)  # shorter edges colored strongly
    return gobj.Scatter3d(
        x=ex,y=ey,z=ez,
        mode='lines',
        opacity=edges_opacity,
        text=edge_hovertext,
        name="edges",
        line=dict(
            width=4,
            #color=edge_colors,
            color=edge_dist,
            reversescale=False,
            colorscale="Hot",
            colorbar=dict(
                title="edge distance (m)",
                len=0.8),
            showscale=True
        ),
    )

@torch.inference_mode()
def plot_nodes(_graph: Data) -> gobj.Scatter3d:
    graph = copy.deepcopy(_graph)
    inference_graph = transform(_graph)
    GCN.eval()
    pred = GCN.forward(inference_graph.x, inference_graph.edge_index, inference_graph.edge_attr)
    pred = torch.sigmoid(pred)
    state_var, pos_var, existence_mask = pred.transpose(1, 0)
    x, y, z = graph.pos.transpose(1,0)
    nodes_hovertext = []
    # color = torch.abs(graph.y[:, 0] - state_var)
    thresh = 0.75
    pred_change_mask = state_var > thresh
    actual_change_mask = graph.y[:,0] > thresh
    color = ["rgb(0,0,0)" for _ in range(len(pred_change_mask))]
    for ind, (pred, actual) in enumerate(zip(pred_change_mask, actual_change_mask)):
        if pred == actual:
            color[ind] = "rgb(0, 255,0)"
        else:
            if pred:
                color[ind] = "rgb(0,0,255)" # blue false positive
            if actual:
                color[ind] = "rgb(255,0,0)"  # red false neg
    for cls, attr, gt, s, p, e in zip(graph.classifications, graph.x, graph.y, state_var, pos_var, existence_mask):
        inference_result = nl.join([f"state variability: {s:.3f}", f"position variability: {p:.3f}", f"existence mask: {e:.3f}"])
        nodes_hovertext.append(nl.join(
            [str(Objects3DSSG(int(cls))), inference_result, node_attr_to_str(attr)]))
    nodes_trace = gobj.Scatter3d(
        x=x,y=y,z=z,
        text=nodes_hovertext,
        mode='markers',
        name="nodes",
        marker=dict(
            showscale=True,
            colorscale="Hot",
            color=color,
            size=10*(1+pos_var),
        ),
    )
    return nodes_trace


def visualize_one_graph(root: str, graph: Data, _plot_mesh=True):
    if not graph:
        return None
    load_centroids(root, graph)
    pl_nodes = plot_nodes(graph)
    _plots = [pl_nodes]
    if edges_opacity > 0.2:
        pl_edges = plot_edges(graph)
        _plots.append(pl_edges)
    pl_mesh = plot_mesh(root, graph) if _plot_mesh else None
    if pl_mesh is not None:
        _plots.append(pl_mesh)
    fig = gobj.Figure(
        data=_plots,
    )
    fig.update_scenes({"aspectmode":"data"})  # xyz on same scale
    return fig


def load_dataset():
    dataset = SceneGraphChangeDataset()
    global root
    root = dataset.root
    return dataset


def dash_app(dataset, scan_id_to_idx):
    all_scans = list(scan_id_to_idx.keys())
    from dash import Dash, dcc, html, Input, Output
    import dash_bootstrap_components as dbc
    global graph_stats_str, curr_refscan_id, ref_scan_to_rescan, curr_rescans

    app = Dash(
        name="SceneChangeDataset visualization",
        external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'],
    )
    def _serve_layout():
        global graph_stats_str, curr_refscan_id, ref_scan_to_rescan, curr_rescans
        ret = html.Div([
        html.Div([
            html.Div([
                html.P("Edges opacity"),
                dcc.Slider(
                    id='edges_opacity',
                    min=0.0,
                    max=1.0,
                    value=1.0,
                ),
                html.P("Mesh opacity"),
                dcc.Slider(
                    id='mesh_opacity',
                    min=0.0,
                    max=1.0,
                    value=1.0,
                ),
                html.P("Edges connectivity %"),
                dcc.Slider(25, 100, 25,
                    id='edges_connectivity',
                    value=100,
                ),
                html.P("refscan_id:"),
                dcc.Dropdown(
                    options=all_ref_scans,
                    value=curr_refscan_id,
                    id='refscan_id'
                ),
                html.P("rescan_id:"),
                dcc.Dropdown(
                    options=curr_rescans,
                    value=curr_rescans[0],
                    id='rescan_id',
                ),
                html.P("graph stats:"),
                dcc.Textarea(
                    readOnly=True,
                    id="graph_stats",
                    style = {"width":"350px", "height":"350px"},
                )
            ], style={"width": "400px", "margin-left": 0}, className='six columns'),

        html.Div([
            dcc.Graph(
                id='graph_vis',
                style={'width': '120vh', 'height': '120vh'},
            ),
        ], className="six columns"),
        ], className="row")
        ])
        return ret

    app.layout = _serve_layout

    @app.callback(
        Output('graph_vis', 'figure'),
        Output('graph_stats', 'value'),
        Output('rescan_id', 'options'),
        Input('mesh_opacity', 'value'),
        Input('edges_opacity', 'value'),
        Input('rescan_id', 'value'),
        Input('refscan_id', 'value'),
        Input('edges_connectivity', 'value'),
    )
    def update_mesh_opacity(m, e, g, r, ctv):
        global mesh_opacity, edges_opacity, curr_graph, graph_stats_str, dist_filter
        global graph_stats_str, curr_refscan_id, ref_scan_to_rescan, curr_rescans
        curr_refscan_id = r
        curr_rescans = ref_scan_to_rescan[r]
        mesh_opacity = m
        edges_opacity = e
        ctv_lvl: int = min(int(ctv / 25), 4)
        print(ctv_lvl)
        thresh = connectivity_to_dist[ctv_lvl]
        print(thresh)
        dist_filter.abs_distance_threshold = thresh
        print(f"using {dist_filter.abs_distance_threshold=} m")

        curr_graph = dist_filter(dataset[scan_id_to_idx[g]])

        graph_stats_str = summarize_graph(curr_graph)
        _plot_mesh = mesh_opacity > 0.05
        print(curr_graph)
        return visualize_one_graph(root, curr_graph, _plot_mesh), graph_stats_str, curr_rescans

    return app


if __name__ == "__main__":
    # run at top level dir with python ./visualize_graph.py config.gin
    import sys
    import gin
    config_files = sys.argv[1:]
    gin.parse_config_files_and_bindings(config_files, "", skip_unknown=True)
    dataset = load_dataset()
    dataset._load_raw_files()
    # 120
    _GCN = torch.load("pca_model_final.pt")
    GCN = SimpleMPGNN(dataset.num_node_features, dataset.num_classes, dataset.num_edge_features, [16])
    GCN.load_state_dict(_GCN.state_dict())
    transform = dataset.transform
    dataset.transform = None;
    for one_scan in dataset.scans:
        refscan_id = one_scan["reference"]
        rescans = [s["reference"] for s in one_scan["scans"]]
        ref_scan_to_rescan[refscan_id] = rescans
        print(f"{refscan_id=} has {len(rescans)} rescans")
    print(len(dataset.scans))
    load_centroids(dataset.root, dataset[0])
    scan_id_to_idx = {d.input_graph: idx for idx, d in enumerate(dataset)}
    all_ref_scans =list(ref_scan_to_rescan.keys())
    curr_rescans = ref_scan_to_rescan[curr_refscan_id]
    app = dash_app(dataset, scan_id_to_idx)
    app.run_server(port="8051",debug=True)

