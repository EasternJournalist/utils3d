from io import TextIOWrapper
from typing import Any, Union


def read_obj(file : Any, encoding: Union[str, None] = None):
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
            'mtllib': [],
            'v': [[0,1, 0.2, 1.0], [1.2, 0.0, 0.0], ...],
            'vt': [[0.5, 0.5], ...],
            'vn': [[0., 0.7, 0.7], [0., -0.7, 0.7], ...],
            'f': [[[1, 0, 0], [2, 0, 0], [3, 0, 0]], ...]   # index in the order of (face, vertex, v/vt/vn). NOTE: Indices start from 1. 0 indicates skip.
            'usemtl': [{'name': 'mtl1', 'f': 7}]
        }
    """
    if isinstance(file, TextIOWrapper):
        lines = file.readlines()
    else:
        with open(file, 'r', encoding=encoding) as fp:
            lines = fp.readlines()
    mtllib = []
    v = []
    vt = []
    vn = []
    vp = []
    f = []
    usemtl = []

    pad0 = lambda l: l + [0] * (3 - len(l))

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
            f.append([pad0([int(i) if i else 0 for i in e.split('/')]) for e in s[1:]])
        elif s[0] == 'usemtl':
            assert len(s) == 2
            usemtl.append({'name': s[1], 'f':len(f)})
        elif s[0] == 'mtllib':
            assert len(s) == 2
            mtllib.append(s[1])
        elif s[0][0] == '#':
            continue
        else:
            raise Exception()
    
    return {
        'mtllib': mtllib,
        'v': v,
        'vt': vt,
        'vn': vn,
        'vp': vp,
        'f': f,
        'usemtl': usemtl,
    }
