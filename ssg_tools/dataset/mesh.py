import trimesh
from typing import Literal, Optional
import numpy as np
import numpy.lib.recfunctions as rcf
import trimesh.visual
from ssg_tools.dataset.structured_array import join_record_arrays

def mesh_get_labels(mesh: trimesh.Trimesh, label_type: Literal["segment", "nyu40", "eigen13", "rio27"] = "segment", point_element: str = "vertex") -> np.ndarray:
    """Access the labels of the given label type.
    Args:
        label_type: The type of the labels to access. Defaults to "segment", which loads the standard labels.

    Raises:
        ValueError: If an unsupported label type is given.

    Returns:
        The loaded labels.
    """

    
    label_type = label_type.lower()
    point_data = mesh.metadata['_ply_raw'][point_element]['data']

    if label_type == 'segment':
        try:
            labels = point_data['objectId']
        except:
            labels = point_data['label']
    elif label_type == 'nyu40':
        labels = point_data['NYU40']
    elif label_type == 'eigen13':
        labels = point_data['Eigen13']
    elif label_type == 'rio27':
        labels = point_data['RIO27']
    else:
        raise ValueError('unsupported label type:', label_type)
    return labels

def mesh_labels_to_attributes(mesh: trimesh.Trimesh, label_type: Literal["segment", "nyu40", "eigen13", "rio27"] = "segment", point_element: str = "vertex") -> np.ndarray:

    labels = mesh_get_labels(mesh, label_type, point_element)
    mesh.vertex_attributes["labels"] = labels


def preprocess_mesh(mesh, label_type: Optional[Literal["segment", "nyu40", "eigen13", "rio27"]] = None, point_element: str = "vertex"):
    if not hasattr(mesh, "vertex_attributes"):
        mesh.vertex_attributes = {}
    if label_type:
        labels = mesh_get_labels(mesh, label_type, point_element)
        mesh.vertex_attributes["labels"] = labels
    # load colors into the mesh
    if hasattr(mesh, "visual") and isinstance(mesh.visual, trimesh.visual.TextureVisuals):
        mesh.visual = mesh.visual.to_color()
        mesh.vertex_attributes["colors"] = mesh.visual.vertex_colors
    else:
        colors = mesh_get_colors(mesh, point_element=point_element)
        #mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=colors)
        mesh.vertex_attributes["colors"] = colors #np.insert(colors, 3, 255, axis=1)
    
    # load normals or determine them automatically
    if hasattr(mesh, "faces"):
        mesh.vertex_attributes["normals"] = mesh.vertex_normals
    else:
        normals = mesh_get_normals(mesh, point_element=point_element)
        mesh.vertex_attributes["normals"] = normals
    return mesh

#def preprocess_point_cloud(point_cloud, point_element: str = "vertex"):
#    labels = mesh_get_labels(point_cloud, label_type="segment", point_element=point_element)
#    point_cloud.vertex_attributes = {"labels": labels}
#    return point_cloud

def _read_stacked(point_data, 
                  fields: tuple[str, ...]):
    """Helper method to load and stack a set of fields.
    """
    return np.stack([point_data[f] for f in fields], axis=1)


def mesh_get_colors(mesh: trimesh.Trimesh, point_element: str = 'vertex'):
    point_data = mesh.metadata['_ply_raw'][point_element]['data']
    return _read_stacked(point_data, ("red", "green", "blue"))


def mesh_get_normals(mesh: trimesh.Trimesh, point_element: str = 'vertex'):
    point_data = mesh.metadata['_ply_raw'][point_element]['data']
    return _read_stacked(point_data, ("nx", "ny", "nz"))

def mesh_set_labels(mesh: trimesh.Trimesh, labels: np.ndarray, label_type: Literal["segment", "nyu40", "eigen13", "rio27"] = "segment"):
    if label_type == 'segment':
        label_name = 'label'
    elif label_type == 'nyu40':
        label_name = "NYU40"
    elif label_type == "eigen13":
        label_name = "Eigen13"
    elif label_type == "rio27":
        label_name = "RIO27"
    else:
        raise ValueError('unsupported label type:', label_type)
    
    mesh.vertex_attributes[label_name] = labels.squeeze()


def get_stacked_points(data: trimesh.Trimesh, load_colors: bool = False, load_normals: bool = False) -> np.ndarray:
    points = data.vertices.astype(np.float32).view([("points", np.float32, 3)])
    to_join = [points]
    if load_colors:
        colors = data.vertex_attributes["colors"][:, :3]#
        # scale to [-1.0, 1.0]
        colors = (colors.astype(np.float32) / 255 * 2.0 - 1).view([("colors", np.float32, 3)])
        to_join.append(colors)
    if load_normals:
        normals = data.vertex_attributes["normals"]
        normals = normals.astype(np.float32).view([("normals", np.float32, 3)])
        to_join.append(normals)
    points = join_record_arrays(to_join)
    return points