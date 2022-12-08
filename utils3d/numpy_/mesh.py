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
    index, source = index[i_sort], source[i_sort]
    uni, uni_i = np.unique(index, return_index=True)
    input[(slice(None),)*(axis % len(input.shape)) + (uni,)] += np.add.reduceat(source, uni_i, axis)
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
    face_normal = np.repeat(face_normal[..., None, :], 3, -2).reshape((*face_normal.shape[:-2], -1, 3))
    indices = faces.reshape((-1,))
    vertex_normal = np.zeros_like(vertices)
    vertex_normal = index_add_(vertex_normal, axis=-2, index=indices, source=face_normal)
    vertex_normal = np.nan_to_num(vertex_normal / np.linalg.norm(vertex_normal, axis=-1, keepdims=True))
    return vertex_normal

def merge_duplicate_vertices(vertices: np.ndarray, faces: np.ndarray, tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """Merge duplicate vertices of a triangular mesh. 
    Duplicate vertices are merged by selecte one of them, and the face indices are updated accordingly.

    Args:
        vertices (np.ndarray): 3-dimensional vertices of shape (N, 3)
        faces (np.ndarray): triangular face indices of shape (T, 3)
        tol (float, optional): tolerance for merging. Defaults to 1e-6.

    Returns:
        vertices (np.ndarray): 3-dimensional vertices of shape (N, 3)
        faces (np.ndarray): triangular face indices of shape (T, 3)
    """
    vertices_round = np.round(vertices / tol) * tol
    _, uni_i, uni_inv = np.unique(vertices_round, return_index=True, return_inverse=True, axis=0)
    vertices = vertices[uni_i]
    faces = uni_inv[faces]
    return vertices, faces

def subdivide_mesh_simple(vertices: np.ndarray, faces: np.ndarray, n: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Subdivide a triangular mesh by splitting each triangle into 4 smaller triangles.
    NOTE: All original vertices are kept, and new vertices are appended to the end of the vertex list.
    
    Args:
        vertices (np.ndarray): 3-dimensional vertices of shape (N, 3)
        faces (np.ndarray): triangular face indices of shape (T, 3)
        n (int, optional): number of subdivisions. Defaults to 1.

    Returns:
        vertices (np.ndarray): 3-dimensional vertices of shape (N + ?, 3)
        faces (np.ndarray): triangular face indices of shape (4 * T, 3)
    """
    for _ in range(n):
        edges = np.stack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
        edges = np.sort(edges, axis=2)
        uni_edges, uni_inv = np.unique(edges.reshape(-1, 2), return_inverse=True, axis=0)
        uni_inv = uni_inv.reshape(3, -1)
        midpoints = (vertices[uni_edges[:, 0]] + vertices[uni_edges[:, 1]]) / 2

        n_vertices = vertices.shape[0]
        vertices = np.concatenate([vertices, midpoints], axis=0)
        faces = np.concatenate([
            np.stack([faces[:, 0], n_vertices + uni_inv[0], n_vertices + uni_inv[2]], axis=1),
            np.stack([faces[:, 1], n_vertices + uni_inv[1], n_vertices + uni_inv[0]], axis=1),
            np.stack([faces[:, 2], n_vertices + uni_inv[2], n_vertices + uni_inv[1]], axis=1),
            np.stack([n_vertices + uni_inv[0], n_vertices + uni_inv[1], n_vertices + uni_inv[2]], axis=1),
        ], axis=0)
    return vertices, faces