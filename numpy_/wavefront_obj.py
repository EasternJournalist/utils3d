from io import TextIOWrapper
from typing import Any, Union

def read_obj(file : Any, encoding: str = ...):
    """Read wavefront .obj file, without preprocessing.
    
    Why bothering having this read_obj() while we already have other libraries like `trimesh`? 
    This function read the raw format from .obj file and keeps the order of vertices and faces, 
    while trimesh which involves modification like merge/split vertices, which could break the orders of vertices and faces,
    Those libraries are commonly aiming at geometry processing and rendering supporting various formats.
    If you want mesh geometry processing, you may turn to `trimesh` for more features.

    Args:
        file (Any): filepath
        encoding (str, optional): 
    
    Returns:
        obj (dict): A dict containing .obj components
        {
            'v': [[0,1, 0.2, 1.0], [1.2, 0.0, 0.0], ...],
            'vt': [[0.5, 0.5], ...],
            'vn'
            'f': [
                {
                    'usemtl': 'mtl1' ,
                    'v': [[1, 2, 3], [2, 3, 4, 5], ...],
                    'vt': [[4, 2, 3], [1, 7, 8, 9], ...],
                    'vn': [[1, 2, 3], [2, 3, 4, 5], ...]
                }
            ]
        }
    """
    if isinstance(file, TextIOWrapper):
        lines = file.readlines()
    else:
        with open(file, 'r', encoding=encoding) as fp:
            lines = fp.readlines()
    v = []
    vt = []
    vn = []
    vp = []
    faces = []

    current_matname = None
    current_faces_v = []
    current_faces_v = []
    current_faces_v = []

    for line in lines:
        s = line.strip().split()
        if s[0] == 'v':
            assert 4 <= len(s) <= 5
            v.append([float(e) for e in s[1:]])
        elif s[0] == 'vt':
            assert 2 <= len(s) <= 4
            vt.append([float(e) for e in s[1:]])
        elif s[0] == 'vn':
            assert len(s) == 4
            vn.append([float(e) for e in s[1:]])
        elif s[0] == 'vp':
            assert 2 <= len(s) <= 4
            vp.append([float(e) for e in s[1:]])
        elif s[0] == 'f':
            t = [e.split('/') for e in s[1:]]
            len(t[0]) 
        elif s[0] == 'usemtl':
            assert len(s) == 2
            current_matname = s[1]
        elif s[0][0] == '#':
            continue
        else:
            raise Exception()
    
    obj = {}

    