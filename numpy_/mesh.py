import numpy as np
from typing import Tuple

def triangulate(faces: np.ndarray) -> np.ndarray:
    assert len(faces.shape) == 2
    if faces.shape[1] == 3:
        return faces
    n = faces.shape[1]
    loop_indice = np.stack([np.zeros(n - 2, dtype=int), np.arange(1, n - 1, 1, dtype=int), np.arange(2, n, 1, dtype=int)], axis=1)
    return faces[:, loop_indice].reshape(-1, 3)

def compute_face_normal(vertices: np.ndarray, faces: np.ndarray):
    """Compute face normals of a triangular mesh

    Args:
        vertices (np.ndarray):  3-dimensional vertices of shape (N, 3)
        faces (np.ndarray): triangular face indices of shape (T, 3)

    Returns:
        normals (np.ndarray): face normals of shape (T, 3)
    """
    normal = np.cross(vertices[faces[..., 1]] - vertices[faces[..., 0]], vertices[faces[..., 2]] - vertices[faces[..., 0]])
    normal = np.nan_to_num(normal / np.sum(normal ** 2, axis=-1, keepdims=True) ** 0.5)
    return normal

def index_add_(input: np.ndarray, axis: int, index: np.ndarray, source: np.ndarray):
    i_sort = np.argsort(index)
    index = index[i_sort]
    input = input[i_sort]
    uni, uni_i = np.unique(index, return_index=True)
    input[(slice(None),)*axis + (uni,)] += np.add.reduceat(source, uni_i, axis)
    return input

def compute_vertex_normal(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute vertex normals of a triangular mesh by averaging neightboring face normals

    Args:
        vertices (np.ndarray): 3-dimensional vertices of shape (N, 3)
        faces (np.ndarray): triangular face indices of shape (T, 3)

    Returns:
        normals (np.ndarray): vertex normals of shape (N, 3)
    """
    face_normal = compute_face_normal(vertices, faces)
    face_normal = np.repeat(face_normal[..., None, :], 3, -2).reshape((-1, 3))
    indices = faces.reshape((-1,))
    vertex_normal = np.zeros_like(vertices)
    vertex_normal = index_add_(vertex_normal, 0, indices, face_normal)
    # while len(face_normal) > 0:
    #     v_id, f_i = np.unique(face_indices, return_index=True)
    #     vertex_normal[v_id] += face_normal[f_i]
    #     face_normal = np.delete(face_normal, f_i, axis=0)
    #     face_indices = np.delete(face_indices, f_i)
    vertex_normal = np.nan_to_num(vertex_normal / np.sum(vertex_normal ** 2, axis=-1, keepdims=True) ** 0.5)
    return vertex_normal