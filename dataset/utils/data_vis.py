from vedo import Mesh, show
import os
import numpy as np
from pyvis.network import Network


def load_scene_mesh(scene_folder):
    if os.path.isfile(os.path.join(scene_folder, "labels.instances.annotated.v2.ply")):
        mesh_file = os.path.join(scene_folder, "labels.instances.annotated.v2.ply")
    elif os.path.isfile(os.path.join(scene_folder, "mesh.refined.v2.obj")):
        mesh_file = os.path.join(scene_folder, "mesh.refined.v2.obj")
    else:
        print("No Mesh File Available")
        return

    test_mesh = Mesh(mesh_file)

    max_z = np.max(test_mesh.points()[:, 2])
    height = max_z - 0.5
    chopped_mesh = test_mesh.clone().cutWithPlane(origin=(0, 0, height), normal=(0, 0, -1))
    plt = show(chopped_mesh, axes=2, camera={"pos": (-0.1, -0.1, max_z + 15)})
    plt.screenshot("data/scan-images/{}.png".format(scene_folder))
    plt.close()


def visualize_graph(graph, out_folder, id):
    net_vis = Network('500px', '500px')
    net_vis.from_nx(graph)
    net_vis.show(os.path.join(out_folder, '{}.html'.format(id)))
