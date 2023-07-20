from io import TextIOWrapper
from typing import Dict, Any, Union, Iterable
import numpy as np
from pathlib import Path


__all__ = ['read_obj', 'write_obj', 'simple_write_obj']

def read_obj(file : Any, encoding: Union[str, None] = None, ignore_unknown: bool = False):
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
    o = []
    s = []
    usemtl = []

    pad0 = lambda l: l + [0] * (3 - len(l))

    for line in lines:
        sq = line.strip().split()
        if len(sq) == 0: 
            continue
        if sq[0] == 'v':
            assert 4 <= len(sq) <= 5
            v.append([float(e) for e in sq[1:]])
        elif sq[0] == 'vt':
            assert 2 <= len(sq) <= 4
            vt.append([float(e) for e in sq[1:]])
        elif sq[0] == 'vn':
            assert len(sq) == 4
            vn.append([float(e) for e in sq[1:]])
        elif sq[0] == 'vp':
            assert 2 <= len(sq) <= 4
            vp.append([float(e) for e in sq[1:]])
        elif sq[0] == 'f':
            f.append([pad0([int(i) if i else 0 for i in e.split('/')]) for e in sq[1:]])
        elif sq[0] == 'usemtl':
            assert len(sq) == 2
            usemtl.append((sq[1], len(f)))
        elif sq[0] == 'o':
            assert len(sq) == 2
            o.append((sq[1], len(f)))
        elif sq[0] == 's':
            s.append((sq[1], len(f)))
        elif sq[0] == 'mtllib':
            assert len(sq) == 2
            mtllib.append(sq[1])
        elif sq[0][0] == '#':
            continue
        else:
            if not ignore_unknown:
                raise Exception(f'Unknown keyword {sq[0]}')
    
    return {
        'mtllib': mtllib,
        'v': v,
        'vt': vt,
        'vn': vn,
        'vp': vp,
        'f': f,
        'o': o,
        's': s,
        'usemtl': usemtl,
    }

def write_obj(file: Union[str, Path], obj: Dict[str, Any], encoding: Union[str, None] = None):
    with open(file, 'w', encoding=encoding) as fp:
        for k in ['v', 'vt', 'vn', 'vp']:
            if k not in obj:
                continue
            for v in obj[k]:
                print(k, *map(float, v), file=fp)
        for f in obj['f']:
            print('f', *((str('/').join(map(int, i)) if isinstance(int(i), Iterable) else i) for i in f), file=fp)


def simple_write_obj(file: Union[str, Path], vertices: np.ndarray, faces: np.ndarray, encoding: Union[str, None] = None):
    """Write wavefront .obj file, without preprocessing.
    
    Args:
        vertices (np.ndarray): [N, 3]
        faces (np.ndarray): [T, 3]
        file (Any): filepath
        encoding (str, optional): 
    """
    with open(file, 'w', encoding=encoding) as fp:
        for v in vertices:
            print('v', *map(float, v), file=fp)
        for f in faces:
            print('f', *map(int, f + 1), file=fp)