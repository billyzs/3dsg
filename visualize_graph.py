#!/usr/bin/env python3

import os
import json
import itertools
import numpy as np
import meshio
import plotly.graph_objects as gobj
from dataset import *
from torch_geometric.data import Data
from collections.abc import Sequence

ply_file = "labels.instances.annotated.v2.ply"
nl =  '<br>'


mesh_opacity = 0.5
edge_opacity = 0.3
root = ""
curr_graph = None

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
        opacity=edge_opacity,
        text=edge_hovertext,
        name="edges",
        line=dict(
            width=3,
            #color=edge_colors,
            color=edge_dist,
            reversescale=True,
            colorscale="Hot",
            colorbar=dict(
                title="edge distance (m)",
                len=0.8),
            showscale=True),
    )


def plot_nodes(graph: Data) -> gobj.Scatter3d:
    x, y, z = graph.pos.transpose(1,0)
    nodes_hovertext = []
    for cls, attr in zip(graph.classifications, graph.x):
        nodes_hovertext.append(nl.join(
            [str(Objects3DSSG(int(cls))), node_attr_to_str(attr)]))

    nodes_trace = gobj.Scatter3d(
        x=x,y=y,z=z,
        text=nodes_hovertext,
        mode='markers',
        name="nodes",
        marker=dict(
            # showscale=True,
            color='#000',
        ),
    )
    return nodes_trace


def visualize_one_graph(root: str, graph: Data):
    if not graph:
        return None
    pl_nodes = plot_nodes(graph)
    pl_edges = plot_edges(graph)
    pl_mesh = plot_mesh(root, graph)
    _plots = [pl_nodes, pl_edges]
    if pl_mesh:
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
    app = Dash(
        name="SceneChangeDataset visualization",
        external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'],
    )
    app.layout = html.Div([
    html.Div([
        html.Div([
                html.P("Edges opacity"),
                dcc.Slider(
                    id='edges_opacity',
                    min=0.0,
                    max=1.0,
                ),
                html.P("Mesh opacity"),
                dcc.Slider(
                    id='mesh_opacity',
                    min=0.0,
                    max=1.0,
                ),
                html.P("scan_id:"),
                dcc.Dropdown(options=all_scans, value=all_scans[10], id='scan_id'),
            ], style={"width": "400px", "margin-left": 0}, className='six columns'),

        html.Div([
            dcc.Graph(
                id='graph_vis',
                style={'width': '90vh', 'height': '90vh'},
            ),
        ], className="six columns"),
    ], className="row")
    ])

    @app.callback(
        Output('graph_vis', 'figure'),
        Input('mesh_opacity', 'value'),
        Input('edges_opacity', 'value'),
        Input('scan_id', 'value'),
    )
    def update_mesh_opacity(m, e, g):
        global mesh_opacity, edges_opacity, curr_graph
        mesh_opacity = m
        edges_opacity = e
        curr_graph = dataset[scan_id_to_idx[g]]
        return visualize_one_graph(root, curr_graph)

    return app


if __name__ == "__main__":
    # run at top level dir with python ./visualize_graph.py config.gin
    import sys
    import gin
    config_files = sys.argv[1:]
    gin.parse_config_files_and_bindings(config_files, "", skip_unknown=True)
    dataset = load_dataset()
    # plot_edges(dataset[0])
    scan_id_to_idx = {d.input_graph: idx for idx, d in enumerate(dataset)}
    app = dash_app(dataset, scan_id_to_idx)
    app.run_server(debug=True)

