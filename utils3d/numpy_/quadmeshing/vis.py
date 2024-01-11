import numpy as np
from typing import *


def vis_edge_color(
        vertices: np.ndarray,
        edges: np.ndarray,
        edge_colors: np.ndarray,
    ):
    """
    Construct a mesh to visualize the edge colors.

    Args:
        vertices (np.ndarray): [N, 3]
        edges (np.ndarray): [E, 2]
        edge_colors (np.ndarray): [E, 3]

    Returns:
        vertices (np.ndarray): [E*4, 3]
        faces (np.ndarray): [E, 4]
        colors (np.ndarray): [E*4, 3]
    """
    E = edges.shape[0]
    N = vertices.shape[0]

    vertices = np.concatenate([
        vertices[edges],
        vertices[edges[:, ::-1]] + 1e-6
    ], axis=1).reshape(E*4, 3)
    
    faces = np.arange(E*4).reshape(E, 4)

    colors = edge_colors[:, None, :].repeat(4, axis=1).reshape(E*4, 3)

    return {
        'vertices': vertices,
        'faces': faces,
        'colors': colors,
    }


    