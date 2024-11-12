from typing import *
from pathlib import Path

import numpy as np


def write_glb(path: Union[str, Path], vertices: np.ndarray, faces: np.ndarray, vertex_colors: np.ndarray = None,  uv: np.ndarray = None):
    import pygltflib

    has_colors = vertex_colors is not None
    has_uv = uv is not None

    triangles_bytes = faces.astype(np.uint32).flatten().tobytes()
    vertices_bytes = vertices.astype(np.float32).tobytes()
    vertex_colors_bytes = vertex_colors.astype(np.float32).tobytes() if has_colors else None
    uv_bytes = uv.astype(np.float32).tobytes() if has_uv else None


    gltf = pygltflib.GLTF2(
        scene=0,
        scenes=[pygltflib.Scene(nodes=[0])],
        nodes=[pygltflib.Node(mesh=0)],
        meshes=[
            pygltflib.Mesh(
                primitives=[
                    pygltflib.Primitive(
                        attributes=pygltflib.Attributes(
                            POSITION=1, 
                            COLOR_0=2 if has_colors else None, 
                            TEXCOORD_0=2 + has_colors if has_uv else None
                        ), 
                        indices=0
                    )
                ]
            )
        ],
        accessors=list(filter(None, [
            pygltflib.Accessor(     # triangles accessor
                bufferView=0,
                componentType=pygltflib.UNSIGNED_INT,
                count=faces.size,
                type=pygltflib.SCALAR,
                max=[int(faces.max())],
                min=[int(faces.min())],
            ),
            pygltflib.Accessor(     # vertices accessor
                bufferView=1,
                componentType=pygltflib.FLOAT,
                count=len(vertices),
                type=pygltflib.VEC3,
                max=vertices.max(axis=0).tolist(),
                min=vertices.min(axis=0).tolist(),
            ),
            pygltflib.Accessor(     # vertex colors accessor
                bufferView=2,
                componentType=pygltflib.FLOAT,
                count=len(vertices),
                type=pygltflib.VEC3,
                max=vertex_colors.max(axis=0).tolist(),
                min=vertex_colors.min(axis=0).tolist(),
            ) if has_colors else None,
            pygltflib.Accessor(     # uv accessor
                bufferView=3,
                componentType=pygltflib.FLOAT,
                count=len(uv),
                type=pygltflib.VEC2,
                max=uv.max(axis=0).tolist(),
                min=uv.min(axis=0).tolist(),
            ) if has_uv else None,
        ])),
        bufferViews=list(filter(None, [
            pygltflib.BufferView(    # triangles buffer view
                buffer=0,
                byteLength=len(triangles_bytes),
                target=pygltflib.ELEMENT_ARRAY_BUFFER,
            ),
            pygltflib.BufferView(    # vertices buffer view
                buffer=0,
                byteOffset=len(triangles_bytes),
                byteLength=len(vertices_bytes),
                target=pygltflib.ARRAY_BUFFER,
            ),
            pygltflib.BufferView(    # vertex colors buffer view
                buffer=0,
                byteOffset=len(triangles_bytes) + len(vertices_bytes),
                byteLength=len(vertex_colors_bytes),
                target=pygltflib.ARRAY_BUFFER,
            ) if has_colors else None,
            pygltflib.BufferView(    # uv buffer view
                buffer=0,
                byteOffset=len(triangles_bytes) + len(vertices_bytes) + (len(vertex_colors_bytes) if has_colors else 0),
                byteLength=len(uv_bytes),
                target=pygltflib.ARRAY_BUFFER,
            ) if has_uv else None,
        ])),
        buffers=[
            pygltflib.Buffer(
                byteLength=len(triangles_bytes) + len(vertices_bytes) + (len(vertex_colors_bytes) if has_colors else 0) + (len(uv_bytes) if has_uv else 0),
            )
        ]
    )
    gltf.set_binary_blob(triangles_bytes + vertices_bytes + (vertex_colors_bytes or b'') + (uv_bytes or b''))
    with open(path, 'wb') as f:
        for chunk in gltf.save_to_bytes():
            f.write(chunk)