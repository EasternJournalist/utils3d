import numpy as np
import plyfile
from typing import *
from pathlib import Path


def read_ply(
        file: Union[str, Path],
        encoding: Union[str, None] = None,
        ignore_unknown: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read .ply file, without preprocessing.
    
    Args:
        file (Any): filepath
        encoding (str, optional): 
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: vertices, faces
    """
    plydata = plyfile.PlyData.read(file)
    return np.stack([plydata['vertex'][k] for k in ['x', 'y', 'z']], axis=-1), plydata['face']['vertex_indices']


def write_ply(
        file: Union[str, Path],
        vertices: np.ndarray,
        faces: np.ndarray,
        color: np.ndarray = None,
    ):
    """
    Write .ply file, without preprocessing.
    
    Args:
        file (Any): filepath
        vertices (np.ndarray): [N, 3]
        faces (np.ndarray): [T, 3]
        color (np.ndarray, optional): [N, 3]. Defaults to None.
    """
    assert vertices.ndim == 2 and vertices.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3
    vertices = vertices.astype(np.float32)
    faces = faces.astype(np.int32)
    if color is not None:
        assert color.ndim == 2 and color.shape[1] == 3
        if color.dtype in [np.float32, np.float64]:
            color = color * 255
        color = np.clip(color, 0, 255).astype(np.uint8)
        vertices_data = np.zeros(len(vertices), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        vertices_data['x'] = vertices[:, 0]
        vertices_data['y'] = vertices[:, 1]
        vertices_data['z'] = vertices[:, 2]
        vertices_data['red'] = color[:, 0]
        vertices_data['green'] = color[:, 1]
        vertices_data['blue'] = color[:, 2]
    else:
        vertices_data = np.array([tuple(v) for v in vertices], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    faces_data = np.zeros(len(faces), dtype=[('vertex_indices', 'i4', (3,))])
    faces_data['vertex_indices'] = faces
    plydata = plyfile.PlyData(
        [
            plyfile.PlyElement.describe(vertices_data, 'vertex'),
            plyfile.PlyElement.describe(faces_data, 'face')
        ]
    )
    plydata.write(file)
    