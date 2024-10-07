from typing import *
from pathlib import Path

import numpy as np


def write_glb(path: Union[str, Path], vertices: np.ndarray, faces: np.ndarray, uv: np.ndarray):
    import pygltflib

    triangles_bytes = faces.astype(np.uint32).flatten().tobytes()
    vertices_bytes = vertices.astype(np.float32).tobytes()
    uv_bytes = uv.astype(np.float32).tobytes()

    gltf = pygltflib.GLTF2(
        scene=0,
        scenes=[pygltflib.Scene(nodes=[0])],
        nodes=[pygltflib.Node(mesh=0)],
        meshes=[
            pygltflib.Mesh(
                primitives=[
                    pygltflib.Primitive(
                        attributes=pygltflib.Attributes(POSITION=1, TEXCOORD_0=2), indices=0
                    )
                ]
            )
        ],
        accessors=[
            pygltflib.Accessor(
                bufferView=0,
                componentType=pygltflib.UNSIGNED_INT,
                count=faces.size,
                type=pygltflib.SCALAR,
                max=[int(faces.max())],
                min=[int(faces.min())],
            ),
            pygltflib.Accessor(
                bufferView=1,
                componentType=pygltflib.FLOAT,
                count=len(vertices),
                type=pygltflib.VEC3,
                max=vertices.max(axis=0).tolist(),
                min=vertices.min(axis=0).tolist(),
            ),
            pygltflib.Accessor(
                bufferView=2,
                componentType=pygltflib.FLOAT,
                count=len(uv),
                type=pygltflib.VEC2,
                max=uv.max(axis=0).tolist(),
                min=uv.min(axis=0).tolist(),
            ),
        ],
        bufferViews=[
            pygltflib.BufferView(
                buffer=0,
                byteLength=len(triangles_bytes),
                target=pygltflib.ELEMENT_ARRAY_BUFFER,
            ),
            pygltflib.BufferView(
                buffer=0,
                byteOffset=len(triangles_bytes),
                byteLength=len(vertices_bytes),
                target=pygltflib.ARRAY_BUFFER,
            ),
            pygltflib.BufferView(
                buffer=0,
                byteOffset=len(triangles_bytes) + len(vertices_bytes),
                byteLength=len(uv_bytes),
                target=pygltflib.ARRAY_BUFFER,
            ), 
        ],
        buffers=[
            pygltflib.Buffer(
                byteLength=len(triangles_bytes) + len(vertices_bytes) + len(uv_bytes),
            )
        ]
    )
    gltf.set_binary_blob(triangles_bytes + vertices_bytes + uv_bytes)
    with open(path, 'wb') as f:
        for chunk in gltf.save_to_bytes():
            f.write(chunk)