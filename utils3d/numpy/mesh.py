import numpy as np
from typing import *

from .transforms import unproject_cv, angle_between

__all__ = [
    'triangulate_mesh',
    'compute_face_normals',
    'compute_face_corner_angles',
    'compute_face_corner_normals',
    'compute_vertex_normals',
    'remove_corrupted_faces',
    'merge_duplicate_vertices',
    'remove_unused_vertices',
    'subdivide_mesh',
    'mesh_relations',
    'flatten_mesh_indices',
    'cube',
    'icosahedron',
    'square',
    'camera_frustum',
    'merge_meshes',
    'calc_quad_candidates',
    'calc_quad_distortion',
    'calc_quad_direction',
    'calc_quad_smoothness',
    'solve_quad',
    'solve_quad_qp',
    'tri_to_quad'
]


def triangulate_mesh(
    faces: np.ndarray,
    vertices: np.ndarray = None,
    method: Literal['fan', 'strip', 'diagonal'] = 'fan'
) -> np.ndarray:
    """
    Triangulate a polygonal mesh.

    ## Parameters
        faces (np.ndarray): [L, P] polygonal faces
        vertices (np.ndarray, optional): [N, 3] 3-dimensional vertices.
            If given, the triangulation is performed according to the distance
            between vertices. Defaults to None.
        backslash (np.ndarray, optional): [L] boolean array indicating
            how to triangulate the quad faces. Defaults to None.

    ## Returns
        (np.ndarray): [L * (P - 2), 3] triangular faces
    """
    if faces.shape[-1] == 3:
        return faces
    P = faces.shape[-1]
    if method == 'fan':
        i = np.arange(P - 2, dtype=int)
        loop_indices = np.stack([np.zeros_like(i), i + 1, i + 2], axis=1)
        return faces[:, loop_indices].reshape((-1, 3))
    elif method == 'strip':
        i = np.arange(P - 2, dtype=int)
        j = i // 2
        loop_indices = np.where(
            (i % 2 == 0)[:, None],
            np.stack([(P - j) % P, j + 1, P - j - 1], axis=1),
            np.stack([j + 1, j + 2, P - j - 1], axis=1)
        )
        return faces[:, loop_indices].reshape((-1, 3))
    elif method == 'diagonal':
        assert faces.shape[-1] == 4, "Diagonal-aware method is only supported for quad faces"
        assert vertices is not None, "Vertices must be provided for diagonal method"
        backslash = np.linalg.norm(vertices[faces[:, 0]] - vertices[faces[:, 2]], axis=-1) < \
                        np.linalg.norm(vertices[faces[:, 1]] - vertices[faces[:, 3]], axis=-1)
        faces = np.where(
            backslash[:, None],
            faces[:, [0, 1, 2, 0, 2, 3]],
            faces[:, [0, 1, 3, 3, 1, 2]]
        ).reshape((-1, 3))
        return faces


def compute_face_corner_angles(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    """
    Compute face corner angles of a mesh

    ## Parameters
        vertices (np.ndarray): [..., N, 3] vertices
        faces (np.ndarray): [T, P] face vertex indices, where P is the number of vertices per face

    ## Returns
        angles (np.ndarray): [..., T, P] face corner angles
    """
    loop = np.arange(faces.shape[1])
    edges = vertices[..., faces[:, np.roll(loop, -1)], :] - vertices[..., faces[:, loop], :]
    angles = angle_between(-np.roll(edges, 1, axis=-2), edges)
    return angles


def compute_face_corner_normals(
    vertices: np.ndarray,
    faces: np.ndarray,
    normalized: bool = True
) -> np.ndarray:
    """
    Compute the face corner normals of a mesh

    ## Parameters
        vertices (np.ndarray): [..., N, 3] vertices
        faces (np.ndarray): [T, P] face vertex indices, where P is the number of vertices per face

    ## Returns
        angles (np.ndarray): [..., T, P, 3] face corner normals
    """
    loop = np.arange(faces.shape[1])
    edges = vertices[..., faces[:, np.roll(loop, -1)], :] - vertices[..., faces[:, loop], :]
    normals = np.cross(np.roll(edges, 1, axis=-2), edges)
    if normalized:
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True) + np.finfo(vertices.dtype).eps
    return normals


def compute_face_normals(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    """
    Compute face normals of a mesh

    ## Parameters
        vertices (np.ndarray): [..., N, 3] 3-dimensional vertices
        faces (np.ndarray): [T, P] face indices

    ## Returns
        normals (np.ndarray): [..., T, 3] face normals
    """
    if faces.shape[-1] == 3:
        normals = np.cross(
            vertices[..., faces[:, 1], :] - vertices[..., faces[:, 0], :],
            vertices[..., faces[:, 2], :] - vertices[..., faces[:, 0], :]
        )
    else:
        normals = compute_face_corner_normals(vertices, faces, normalized=False)
        normals = np.mean(normals, axis=-2)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True) + np.finfo(vertices.dtype).eps
    return normal


def compute_vertex_normals(
    vertices: np.ndarray,
    faces: np.ndarray,
    weighted: Literal['uniform', 'area', 'angle'] = 'uniform'
) -> np.ndarray:
    """
    Compute vertex normals of a triangular mesh by averaging neighboring face normals

    ## Parameters
        vertices (np.ndarray): [..., N, 3] 3-dimensional vertices
        faces (np.ndarray): [T, P] face vertex indices, where P is the number of vertices per face

    ## Returns
        normals (np.ndarray): [..., N, 3] vertex normals (already normalized to unit vectors)
    """
    face_corner_normals = compute_face_corner_normals(vertices, faces, normalized=False)
    if weighted == 'uniform':
        face_corner_normals /= np.linalg.norm(face_corner_normals, axis=-1, keepdims=True) + np.finfo(vertices.dtype).eps
    elif weighted == 'area':
        pass
    elif weighted == 'angle':
        face_corner_angle = compute_face_corner_angles(vertices, faces)
        face_corner_normals *= face_corner_angle[..., None]
    vertex_normals = np.zeros_like(vertices, dtype=vertices.dtype)
    np.add.at(
        vertex_normals, 
        (..., faces[..., None], np.arange(3)), 
        face_corner_normals
    )
    vertex_normals /= np.linalg.norm(vertex_normals, axis=-1, keepdims=True) + np.finfo(vertices.dtype).eps
    return vertex_normals



def remove_corrupted_faces(faces: np.ndarray) -> np.ndarray:
    """
    Remove corrupted faces (faces with duplicated vertices)

    ## Parameters
        faces (np.ndarray): [T, 3] triangular face indices

    ## Returns
        np.ndarray: [T_, 3] triangular face indices
    """
    corrupted = (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 2] == faces[:, 0])
    return faces[~corrupted]


def merge_duplicate_vertices(
    vertices: np.ndarray, 
    faces: np.ndarray,
    tol: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge duplicate vertices of a triangular mesh. 
    Duplicate vertices are merged by selecte one of them, and the face indices are updated accordingly.

    ## Parameters
        vertices (np.ndarray): [N, 3] 3-dimensional vertices
        faces (np.ndarray): [T, 3] triangular face indices
        tol (float, optional): tolerance for merging. Defaults to 1e-6.

    ## Returns
        vertices (np.ndarray): [N_, 3] 3-dimensional vertices
        faces (np.ndarray): [T, 3] triangular face indices
    """
    vertices_round = np.round(vertices / tol)
    _, uni_i, uni_inv = np.unique(vertices_round, return_index=True, return_inverse=True, axis=0)
    vertices = vertices[uni_i]
    faces = uni_inv[faces]
    return vertices, faces


def remove_unused_vertices(
    faces: np.ndarray,
    *vertice_attrs,
    return_indices: bool = False
) -> Tuple[np.ndarray, ...]:
    """
    Remove unreferenced vertices of a mesh. 
    Unreferenced vertices are removed, and the face indices are updated accordingly.

    ## Parameters
        faces (np.ndarray): [T, P] face indices
        *vertice_attrs: vertex attributes

    ## Returns
        faces (np.ndarray): [T, P] face indices
        *vertice_attrs: vertex attributes
        indices (np.ndarray, optional): [N] indices of vertices that are kept. Defaults to None.
    """
    P = faces.shape[-1]
    fewer_indices, inv_map = np.unique(faces, return_inverse=True)
    faces = inv_map.astype(np.int32).reshape(-1, P)
    ret = [faces]
    for attr in vertice_attrs:
        ret.append(attr[fewer_indices])
    if return_indices:
        ret.append(fewer_indices)
    return tuple(ret)


def subdivide_mesh(
    vertices: np.ndarray,
    faces: np.ndarray, 
    n: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Subdivide a triangular mesh by splitting each triangle into 4 smaller triangles.
    NOTE: All original vertices are kept, and new vertices are appended to the end of the vertex list.
    
    ## Parameters
        vertices (np.ndarray): [N, 3] 3-dimensional vertices
        faces (np.ndarray): [T, 3] triangular face indices
        n (int, optional): number of subdivisions. Defaults to 1.

    ## Returns
        vertices (np.ndarray): [N_, 3] subdivided 3-dimensional vertices
        faces (np.ndarray): [4 * T, 3] subdivided triangular face indices
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


def mesh_relations(
    faces: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the relation between vertices and faces.
    NOTE: The input mesh must be a manifold triangle mesh.

    ## Parameters
        faces (np.ndarray): [T, 3] triangular face indices

    ## Returns
        edges (np.ndarray): [E, 2] edge indices
        edge2face (np.ndarray): [E, 2] edge to face relation. The second column is -1 if the edge is boundary.
        face2edge (np.ndarray): [T, 3] face to edge relation
        face2face (np.ndarray): [T, 3] face to face relation
    """
    T = faces.shape[0]
    edges = np.stack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=1).reshape(-1, 2)  # [3T, 2]
    edges = np.sort(edges, axis=1)  # [3T, 2]
    edges, face2edge, occurence = np.unique(edges, axis=0, return_inverse=True, return_counts=True) # [E, 2], [3T], [E]
    E = edges.shape[0]
    assert np.all(occurence <= 2), "The input mesh is not a manifold mesh."

    # Edge to face relation
    padding = np.arange(E, dtype=np.int32)[occurence == 1]
    padded_face2edge = np.concatenate([face2edge, padding], axis=0)  # [2E]
    edge2face = np.argsort(padded_face2edge, kind='stable').reshape(-1, 2) // 3  # [E, 2]
    edge2face_valid = edge2face[:, 1] < T   # [E]
    edge2face[~edge2face_valid, 1] = -1

    # Face to edge relation
    face2edge = face2edge.reshape(-1, 3)  # [T, 3]

    # Face to face relation
    face2face = edge2face[face2edge]  # [T, 3, 2]
    face2face = face2face[face2face != np.arange(T)[:, None, None]].reshape(T, 3)  # [T, 3]
    
    return edges, edge2face, face2edge, face2face


@overload
def flatten_mesh_indices(faces1: np.ndarray, attr1: np.ndarray, *other_faces_attrs_pairs: np.ndarray) -> Tuple[np.ndarray, ...]: 
    """
    Rearrange the indices of a mesh to a flattened version. Vertices will be no longer shared.

    ### Parameters:
    - `faces1`: [T, P] face indices of the first attribute
    - `attr1`: [N1, ...] attributes of the first mesh
    - ...

    ### ## Returns
    - `faces`: [T, P] flattened face indices, contigous from 0 to T * P - 1
    - `attr1`: [T * P, ...] attributes of the first mesh, where every P values correspond to a face
    _ ...
    """
def flatten_mesh_indices(*args: np.ndarray) -> Tuple[np.ndarray, ...]:
    assert len(args) % 2 == 0, "The number of arguments must be even."
    T, P = args[0].shape
    assert all(arg.shape[0] == T and arg.shape[1] == P for arg in args[::2]), "The faces must have the same shape."
    attr_flat = []
    for faces_, attr_ in zip(args[::2], args[1::2]):
        attr_flat_ = attr_[faces_].reshape(-1, *attr_.shape[1:])
        attr_flat.append(attr_flat_)
    faces_flat = np.arange(T * P, dtype=np.int32).reshape(T, P)
    return faces_flat, *attr_flat



def square(tri: bool = False) -> Tuple[np.ndarray, np.ndarray]: 
    """
    Get a square mesh of area 1 centered at origin in the xy-plane.

    ### Returns
        vertices (np.ndarray): shape (4, 3)
        faces (np.ndarray): shape (1, 4)
    """
    vertices = np.array([
        [-0.5, 0.5, 0],   [0.5, 0.5, 0],   [0.5, -0.5, 0],   [-0.5, -0.5, 0] # v0-v1-v2-v3
    ], dtype=np.float32)
    if tri:
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    else:
        faces = np.array([[0, 1, 2, 3]], dtype=np.int32)
    return vertices, faces  


def cube(tri: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get x cube mesh of size 1 centered at origin.

    ### Parameters
        tri (bool, optional): return triangulated mesh. Defaults to False, which returns quad mesh.

    ### Returns
        vertices (np.ndarray): shape (8, 3) 
        faces (np.ndarray): shape (12, 3)
    """
    vertices = np.array([
        [0.5, 0.5, 0.5],   [-0.5, 0.5, 0.5],   [-0.5, -0.5, 0.5],   [0.5, -0.5, 0.5], # v0-v1-v2-v3
        [0.5, 0.5, -0.5],  [-0.5, 0.5, -0.5],  [-0.5, -0.5, -0.5],  [0.5, -0.5, -0.5] # v4-v5-v6-v7
    ], dtype=np.float32).reshape((-1, 3))

    faces = np.array([
        [0, 1, 2, 3], #  (front)
        [5, 4, 7, 6], #  (back)
        [4, 5, 1, 0], #  (top)
        [2, 6, 7, 3], #  (bottom)
        [1, 5, 6, 2], #  (left)
        [4, 0, 3, 7], #  (right)
    ], dtype=np.int32)

    if tri:
        faces = triangulate_mesh(faces, vertices=vertices)

    return vertices, faces


def camera_frustum(extrinsics: np.ndarray, intrinsics: np.ndarray, depth: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get x triangle mesh of camera frustum.
    """
    assert extrinsics.shape == (4, 4) and intrinsics.shape == (3, 3)
    vertices = unproject_cv(
        np.array([[0, 0], [0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.float32), 
        np.array([0] + [depth] * 4, dtype=np.float32), 
        intrinsics,
        extrinsics
    ).astype(np.float32)
    edges = np.array([
        [0, 1], [0, 2], [0, 3], [0, 4], 
        [1, 2], [2, 3], [3, 4], [4, 1]
    ], dtype=np.int32)
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 4],
        [0, 4, 1],
        [1, 2, 3],
        [1, 3, 4]
    ], dtype=np.int32)
    return vertices, edges, faces


def icosahedron():
    A = (1 + 5 ** 0.5) / 2
    vertices = np.array([
        [0, 1, A], [0, -1, A], [0, 1, -A], [0, -1, -A],
        [1, A, 0], [-1, A, 0], [1, -A, 0], [-1, -A, 0],
        [A, 0, 1], [A, 0, -1], [-A, 0, 1], [-A, 0, -1]
    ], dtype=np.float32)
    faces = np.array([
        [0, 1, 8], [0, 8, 4], [0, 4, 5], [0, 5, 10], [0, 10, 1],
        [3, 2, 9], [3, 9, 6], [3, 6, 7], [3, 7, 11], [3, 11, 2],
        [1, 6, 8], [8, 9, 4], [4, 2, 5], [5, 11, 10], [10, 7, 1],
        [2, 4, 9], [9, 8, 6], [6, 1, 7], [7, 10, 11], [11, 5, 2]
    ], dtype=np.int32)
    return vertices, faces


def merge_meshes(meshes: List[Tuple[np.ndarray, ...]]) -> Tuple[np.ndarray, ...]:
    """
    Merge multiple meshes into one mesh. Vertices will be no longer shared.

    ### Parameters:
        `meshes`: a list of tuple (faces, vertices_attr1, vertices_attr2, ....)

    ### ## Returns
        `faces`: [sum(T_i), P] merged face indices, contigous from 0 to sum(T_i) * P - 1
        `*vertice_attrs`: [sum(T_i) * P, ...] merged vertex attributes, where every P values correspond to a face
    """
    faces_merged = []
    attrs_merged = [[] for _ in meshes[0][1:]]
    vertex_offset = 0
    for f, *attrs in meshes:
        faces_merged.append(f + vertex_offset)
        vertex_offset += len(attrs[0])
        for attr_merged, attr in zip(attrs_merged, attrs):
            attr_merged.append(attr)
    faces_merged = np.concatenate(faces_merged, axis=0)
    attrs_merged = [np.concatenate(attr_list, axis=0) for attr_list in attrs_merged]
    return (faces_merged, *attrs_merged)



def calc_quad_candidates(
    edges: np.ndarray,
    face2edge: np.ndarray,
    edge2face: np.ndarray,
):
    """
    Calculate the candidate quad faces.

    ## Parameters
        edges (np.ndarray): [E, 2] edge indices
        face2edge (np.ndarray): [T, 3] face to edge relation
        edge2face (np.ndarray): [E, 2] edge to face relation

    ## Returns
        quads (np.ndarray): [Q, 4] quad candidate indices
        quad2edge (np.ndarray): [Q, 4] edge to quad candidate relation
        quad2adj (np.ndarray): [Q, 8] adjacent quad candidates of each quad candidate
        quads_valid (np.ndarray): [E] whether the quad corresponding to the edge is valid
    """
    E = edges.shape[0]
    T = face2edge.shape[0]

    quads_valid = edge2face[:, 1] != -1
    Q = quads_valid.sum()
    quad2face = edge2face[quads_valid]  # [Q, 2]
    quad2edge = face2edge[quad2face]  # [Q, 2, 3]
    flag = quad2edge == np.arange(E)[quads_valid][:, None, None] # [Q, 2, 3]
    flag = flag.argmax(axis=-1)  # [Q, 2]
    quad2edge = np.stack([
        quad2edge[np.arange(Q)[:, None], np.arange(2)[None, :], (flag + 1) % 3],
        quad2edge[np.arange(Q)[:, None], np.arange(2)[None, :], (flag + 2) % 3],
    ], axis=-1).reshape(Q, 4)  # [Q, 4]

    quads = np.concatenate([
        np.where(
            (edges[quad2edge[:, 0:1], 1:] == edges[quad2edge[:, 1:2], :]).any(axis=-1),
            edges[quad2edge[:, 0:1], [[0, 1]]],
            edges[quad2edge[:, 0:1], [[1, 0]]],
        ),
        np.where(
            (edges[quad2edge[:, 2:3], 1:] == edges[quad2edge[:, 3:4], :]).any(axis=-1),
            edges[quad2edge[:, 2:3], [[0, 1]]],
            edges[quad2edge[:, 2:3], [[1, 0]]],
        ),
    ], axis=1)  # [Q, 4]

    quad2adj = edge2face[quad2edge]  # [Q, 4, 2]
    quad2adj = quad2adj[quad2adj != quad2face[:, [0,0,1,1], None]].reshape(Q, 4)  # [Q, 4]
    quad2adj_valid = quad2adj != -1
    quad2adj = face2edge[quad2adj]  # [Q, 4, 3]
    quad2adj[~quad2adj_valid, 0] = quad2edge[~quad2adj_valid]
    quad2adj[~quad2adj_valid, 1:] = -1
    quad2adj = quad2adj[quad2adj != quad2edge[..., None]].reshape(Q, 8)  # [Q, 8]
    edge_valid = -np.ones(E, dtype=np.int32)
    edge_valid[quads_valid] = np.arange(Q)
    quad2adj_valid = quad2adj != -1
    quad2adj[quad2adj_valid] = edge_valid[quad2adj[quad2adj_valid]]  # [Q, 8]

    return quads, quad2edge, quad2adj, quads_valid


def calc_quad_distortion(
    vertices: np.ndarray,
    quads: np.ndarray,
):
    """
    Calculate the distortion of each candidate quad face.

    ## Parameters
        vertices (np.ndarray): [N, 3] 3-dimensional vertices
        quads (np.ndarray): [Q, 4] quad face indices

    ## Returns
        distortion (np.ndarray): [Q] distortion of each quad face
    """
    edge0 = vertices[quads[:, 1]] - vertices[quads[:, 0]]  # [Q, 3]
    edge1 = vertices[quads[:, 2]] - vertices[quads[:, 1]]  # [Q, 3]
    edge2 = vertices[quads[:, 3]] - vertices[quads[:, 2]]  # [Q, 3]
    edge3 = vertices[quads[:, 0]] - vertices[quads[:, 3]]  # [Q, 3]
    cross = vertices[quads[:, 0]] - vertices[quads[:, 2]]  # [Q, 3]

    len0 = np.maximum(np.linalg.norm(edge0, axis=-1), 1e-10)  # [Q]
    len1 = np.maximum(np.linalg.norm(edge1, axis=-1), 1e-10)  # [Q]
    len2 = np.maximum(np.linalg.norm(edge2, axis=-1), 1e-10)  # [Q]
    len3 = np.maximum(np.linalg.norm(edge3, axis=-1), 1e-10)  # [Q]
    len_cross = np.maximum(np.linalg.norm(cross, axis=-1), 1e-10)  # [Q]

    angle0 = np.arccos(np.clip(np.sum(-edge0 * edge1, axis=-1) / (len0 * len1), -1, 1))  # [Q]
    angle1 = np.arccos(np.clip(np.sum(-edge1 * cross, axis=-1) / (len1 * len_cross), -1, 1)) \
           + np.arccos(np.clip(np.sum(cross * edge2, axis=-1) / (len_cross * len2), -1, 1))  # [Q]
    angle2 = np.arccos(np.clip(np.sum(-edge2 * edge3, axis=-1) / (len2 * len3), -1, 1))  # [Q]
    angle3 = np.arccos(np.clip(np.sum(-edge3 * -cross, axis=-1) / (len3 * len_cross), -1, 1)) \
           + np.arccos(np.clip(np.sum(-cross * edge0, axis=-1) / (len_cross * len0), -1, 1))  # [Q]

    normal0 = np.cross(edge0, edge1)  # [Q, 3]
    normal1 = np.cross(edge2, edge3)  # [Q, 3]
    normal0 = normal0 / np.maximum(np.linalg.norm(normal0, axis=-1, keepdims=True), 1e-10)  # [Q, 3]
    normal1 = normal1 / np.maximum(np.linalg.norm(normal1, axis=-1, keepdims=True), 1e-10)  # [Q, 3]
    angle_normal = np.arccos(np.clip(np.sum(normal0 * normal1, axis=-1), -1, 1))  # [Q]

    D90 = np.pi / 2
    D180 = np.pi
    D360 = np.pi * 2
    ang_eng = (np.abs(angle0 - D90)**2 + np.abs(angle1 - D90)**2 + np.abs(angle2 - D90)**2 + np.abs(angle3 - D90)**2) / 4  # [Q]
    dist_eng = np.abs(angle0 - angle2)**2 / np.minimum(np.maximum(np.minimum(angle0, angle2), 1e-10), np.maximum(D180 - np.maximum(angle0, angle2), 1e-10)) \
             + np.abs(angle1 - angle3)**2 / np.minimum(np.maximum(np.minimum(angle1, angle3), 1e-10), np.maximum(D180 - np.maximum(angle1, angle3), 1e-10))  # [Q]
    plane_eng = np.where(angle_normal < D90/2, np.abs(angle_normal)**2, 1e10)  # [Q]
    eng = ang_eng + 2 * dist_eng + 2 * plane_eng  # [Q]

    return eng


def calc_quad_direction(vertices: np.ndarray, quads: np.ndarray):
    """
    Calculate the direction of each candidate quad face.

    ## Parameters
        vertices (np.ndarray): [N, 3] 3-dimensional vertices
        quads (np.ndarray): [Q, 4] quad face indices

    ## Returns
        direction (np.ndarray): [Q, 4] direction of each quad face.
            Represented by the angle between the crossing and each edge.
    """
    mid0 = (vertices[quads[:, 0]] + vertices[quads[:, 1]]) / 2  # [Q, 3]
    mid1 = (vertices[quads[:, 1]] + vertices[quads[:, 2]]) / 2  # [Q, 3]
    mid2 = (vertices[quads[:, 2]] + vertices[quads[:, 3]]) / 2  # [Q, 3]
    mid3 = (vertices[quads[:, 3]] + vertices[quads[:, 0]]) / 2  # [Q, 3]

    cross0 = mid2 - mid0  # [Q, 3]
    cross1 = mid3 - mid1  # [Q, 3]
    cross0 = cross0 / np.maximum(np.linalg.norm(cross0, axis=-1, keepdims=True), 1e-10)  # [Q, 3]
    cross1 = cross1 / np.maximum(np.linalg.norm(cross1, axis=-1, keepdims=True), 1e-10)  # [Q, 3]

    edge0 = vertices[quads[:, 1]] - vertices[quads[:, 0]]  # [Q, 3]
    edge1 = vertices[quads[:, 2]] - vertices[quads[:, 1]]  # [Q, 3]
    edge2 = vertices[quads[:, 3]] - vertices[quads[:, 2]]  # [Q, 3]
    edge3 = vertices[quads[:, 0]] - vertices[quads[:, 3]]  # [Q, 3]
    edge0 = edge0 / np.maximum(np.linalg.norm(edge0, axis=-1, keepdims=True), 1e-10)  # [Q, 3]
    edge1 = edge1 / np.maximum(np.linalg.norm(edge1, axis=-1, keepdims=True), 1e-10)  # [Q, 3]
    edge2 = edge2 / np.maximum(np.linalg.norm(edge2, axis=-1, keepdims=True), 1e-10)  # [Q, 3]
    edge3 = edge3 / np.maximum(np.linalg.norm(edge3, axis=-1, keepdims=True), 1e-10)  # [Q, 3]

    direction = np.stack([
        np.arccos(np.clip(np.sum(cross0 * edge0, axis=-1), -1, 1)),
        np.arccos(np.clip(np.sum(cross1 * edge1, axis=-1), -1, 1)),
        np.arccos(np.clip(np.sum(-cross0 * edge2, axis=-1), -1, 1)),
        np.arccos(np.clip(np.sum(-cross1 * edge3, axis=-1), -1, 1)),
    ], axis=-1)  # [Q, 4]

    return direction


def calc_quad_smoothness(
    quad2edge: np.ndarray,
    quad2adj: np.ndarray,
    quads_direction: np.ndarray,
):
    """
    Calculate the smoothness of each candidate quad face connection.

    ## Parameters
        quad2adj (np.ndarray): [Q, 8] adjacent quad faces of each quad face
        quads_direction (np.ndarray): [Q, 4] direction of each quad face

    ## Returns
        smoothness (np.ndarray): [Q, 8] smoothness of each quad face connection
    """
    Q = quad2adj.shape[0]
    quad2adj_valid = quad2adj != -1
    connections = np.stack([
        np.arange(Q)[:, None].repeat(8, axis=1),
        quad2adj,
    ], axis=-1)[quad2adj_valid]  # [C, 2]
    shared_edge_idx_0 = np.array([[0, 0, 1, 1, 2, 2, 3, 3]]).repeat(Q, axis=0)[quad2adj_valid]  # [C]
    shared_edge_idx_1 = np.argmax(quad2edge[quad2adj][quad2adj_valid] == quad2edge[connections[:, 0], shared_edge_idx_0][:, None], axis=-1)  # [C]
    valid_smoothness = np.abs(quads_direction[connections[:, 0], shared_edge_idx_0] - quads_direction[connections[:, 1], shared_edge_idx_1])**2  # [C]
    smoothness = np.zeros([Q, 8], dtype=np.float32)
    smoothness[quad2adj_valid] = valid_smoothness
    return smoothness


def solve_quad(
    face2edge: np.ndarray,
    edge2face: np.ndarray,
    quad2adj: np.ndarray,
    quads_distortion: np.ndarray,
    quads_smoothness: np.ndarray,
    quads_valid: np.ndarray,
):
    """
    Solve the quad mesh from the candidate quad faces.

    ## Parameters
        face2edge (np.ndarray): [T, 3] face to edge relation
        edge2face (np.ndarray): [E, 2] edge to face relation
        quad2adj (np.ndarray): [Q, 8] adjacent quad faces of each quad face
        quads_distortion (np.ndarray): [Q] distortion of each quad face
        quads_smoothness (np.ndarray): [Q, 8] smoothness of each quad face connection
        quads_valid (np.ndarray): [E] whether the quad corresponding to the edge is valid

    ## Returns
        weights (np.ndarray): [Q] weight of each valid quad face
    """
    import scipy.sparse as sp
    import scipy.optimize as opt

    T = face2edge.shape[0]
    E = edge2face.shape[0]
    Q = quads_distortion.shape[0]
    edge_valid = -np.ones(E, dtype=np.int32)
    edge_valid[quads_valid] = np.arange(Q)

    quads_connection = np.stack([
        np.arange(Q)[:, None].repeat(8, axis=1),
        quad2adj,
    ], axis=-1)[quad2adj != -1]  # [C, 2]
    quads_connection = np.sort(quads_connection, axis=-1)  # [C, 2]
    quads_connection, quads_connection_idx = np.unique(quads_connection, axis=0, return_index=True)  # [C, 2], [C]
    quads_smoothness = quads_smoothness[quad2adj != -1]  # [C]
    quads_smoothness = quads_smoothness[quads_connection_idx]  # [C]
    C = quads_connection.shape[0]

    # Construct the linear programming problem

    # Variables:
    #   quads_weight: [Q] weight of each quad face
    #   tri_min_weight: [T] minimum weight of each triangle face
    #   conn_min_weight: [C] minimum weight of each quad face connection
    #   conn_max_weight: [C] maximum weight of each quad face connection
    # Objective:
    #   mimi

    c = np.concatenate([
        quads_distortion - 3,
        quads_smoothness*4 - 2,
        quads_smoothness*4,
    ], axis=0)  # [Q+C]

    A_ub_triplet = np.concatenate([
        np.stack([np.arange(T), edge_valid[face2edge[:, 0]], np.ones(T)], axis=1),  # [T, 3]
        np.stack([np.arange(T), edge_valid[face2edge[:, 1]], np.ones(T)], axis=1),  # [T, 3]
        np.stack([np.arange(T), edge_valid[face2edge[:, 2]], np.ones(T)], axis=1),  # [T, 3]
        np.stack([np.arange(T, T+C), np.arange(Q, Q+C), np.ones(C)], axis=1),  # [C, 3]
        np.stack([np.arange(T, T+C), quads_connection[:, 0], -np.ones(C)], axis=1),  # [C, 3]
        np.stack([np.arange(T, T+C), quads_connection[:, 1], -np.ones(C)], axis=1),  # [C, 3]
        np.stack([np.arange(T+C, T+2*C), np.arange(Q+C, Q+2*C), -np.ones(C)], axis=1),  # [C, 3]
        np.stack([np.arange(T+C, T+2*C), quads_connection[:, 0], np.ones(C)], axis=1),  # [C, 3]
        np.stack([np.arange(T+C, T+2*C), quads_connection[:, 1], np.ones(C)], axis=1),  # [C, 3]
    ], axis=0)  # [3T+6C, 3]
    A_ub_triplet = A_ub_triplet[A_ub_triplet[:, 1] != -1]  # [3T', 3]
    A_ub = sp.coo_matrix((A_ub_triplet[:, 2], (A_ub_triplet[:, 0], A_ub_triplet[:, 1])), shape=[T+2*C, Q+2*C])  # [T, 
    b_ub = np.concatenate([np.ones(T), -np.ones(C), np.ones(C)], axis=0)  # [T+2C]
    bound = np.stack([
        np.concatenate([np.zeros(Q), -np.ones(C), np.zeros(C)], axis=0),
        np.concatenate([np.ones(Q), np.ones(C), np.ones(C)], axis=0),
    ], axis=1)  # [Q+2C, 2]
    A_eq = None
    b_eq = None

    print('Solver statistics:')
    print(f'    #T = {T}')
    print(f'    #Q = {Q}')
    print(f'    #C = {C}')

    # Solve the linear programming problem
    last_num_valid = 0
    for i in range(100):
        res_ = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bound)
        if not res_.success:
            print(f'    Iter {i} | Failed with {res_.message}')
            break
        res = res_
        weights = res.x[:Q]
        valid = (weights > 0.5)
        num_valid = valid.sum()
        print(f'    Iter {i} | #Q_valid = {num_valid}')
        if num_valid == last_num_valid:
            break
        last_num_valid = num_valid
        A_eq_triplet = np.stack([
            np.arange(num_valid),
            np.arange(Q)[valid],
            np.ones(num_valid),
        ], axis=1)  # [num_valid, 3]
        A_eq = sp.coo_matrix((A_eq_triplet[:, 2], (A_eq_triplet[:, 0], A_eq_triplet[:, 1])), shape=[num_valid, Q+2*C])  # [num_valid, Q+C]
        b_eq = np.where(weights[valid] > 0.5, 1, 0)  # [num_valid]

    # Return the result
    quads_weight = res.x[:Q]
    conn_min_weight = res.x[Q:Q+C]
    conn_max_weight = res.x[Q+C:Q+2*C]
    return quads_weight, conn_min_weight, conn_max_weight


def solve_quad_qp(
    face2edge: np.ndarray,
    edge2face: np.ndarray,
    quad2adj: np.ndarray,
    quads_distortion: np.ndarray,
    quads_smoothness: np.ndarray,
    quads_valid: np.ndarray,
):
    """
    Solve the quad mesh from the candidate quad faces.

    ## Parameters
        face2edge (np.ndarray): [T, 3] face to edge relation
        edge2face (np.ndarray): [E, 2] edge to face relation
        quad2adj (np.ndarray): [Q, 8] adjacent quad faces of each quad face
        quads_distortion (np.ndarray): [Q] distortion of each quad face
        quads_smoothness (np.ndarray): [Q, 8] smoothness of each quad face connection
        quads_valid (np.ndarray): [E] whether the quad corresponding to the edge is valid

    ## Returns
        weights (np.ndarray): [Q] weight of each valid quad face
    """
    import scipy.sparse as sp
    import piqp

    T = face2edge.shape[0]
    E = edge2face.shape[0]
    Q = quads_distortion.shape[0]
    edge_valid = -np.ones(E, dtype=np.int32)
    edge_valid[quads_valid] = np.arange(Q)

    # Construct the quadratic programming problem
    C_smoothness_triplet = np.stack([
        np.arange(Q)[:, None].repeat(8, axis=1)[quad2adj != -1],
        quad2adj[quad2adj != -1],
        5 * quads_smoothness[quad2adj != -1],
    ], axis=-1)  # [C, 3]
    # C_smoothness_triplet = np.concatenate([
    #     C_smoothness_triplet,
    #     np.stack([np.arange(Q), np.arange(Q), 20*np.ones(Q)], axis=1),
    # ], axis=0)  # [C+Q, 3]
    C_smoothness = sp.coo_matrix((C_smoothness_triplet[:, 2], (C_smoothness_triplet[:, 0], C_smoothness_triplet[:, 1])), shape=[Q, Q])  # [Q, Q]
    C_smoothness = C_smoothness.tocsc()
    C_dist = quads_distortion - 20  # [Q]

    A_eq = sp.coo_matrix((np.zeros(Q), (np.zeros(Q), np.arange(Q))), shape=[1, Q])  # [1, Q]\
    A_eq = A_eq.tocsc()
    b_eq = np.array([0])

    A_ub_triplet = np.concatenate([
        np.stack([np.arange(T), edge_valid[face2edge[:, 0]], np.ones(T)], axis=1),  # [T, 3]
        np.stack([np.arange(T), edge_valid[face2edge[:, 1]], np.ones(T)], axis=1),  # [T, 3]
        np.stack([np.arange(T), edge_valid[face2edge[:, 2]], np.ones(T)], axis=1),  # [T, 3]
    ], axis=0)  # [3T, 3]
    A_ub_triplet = A_ub_triplet[A_ub_triplet[:, 1] != -1]  # [3T', 3]
    A_ub = sp.coo_matrix((A_ub_triplet[:, 2], (A_ub_triplet[:, 0], A_ub_triplet[:, 1])), shape=[T, Q])  # [T, Q]
    A_ub = A_ub.tocsc()
    b_ub = np.ones(T)

    lb = np.zeros(Q)
    ub = np.ones(Q)

    solver = piqp.SparseSolver()
    solver.settings.verbose = True
    solver.settings.compute_timings = True
    solver.setup(C_smoothness, C_dist, A_eq, b_eq, A_ub, b_ub, lb, ub)

    status = solver.solve()

    # x = cp.Variable(Q)
    # prob = cp.Problem(
    #     cp.Minimize(cp.quad_form(x, C_smoothness) + C_dist.T @ x),
    #     [
    #         A_ub @ x <= b_ub,
    #         x >= 0, x <= 1,
    #     ]
    # )

    # # Solve the quadratic programming problem
    # prob.solve(solver=cp.PIQP, verbose=True)

    # Return the result
    weights = solver.result.x
    return weights


def tri_to_quad(
    vertices: np.ndarray,
    faces: np.ndarray, 
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a triangle mesh to a quad mesh.
    NOTE: The input mesh must be a manifold mesh.

    ## Parameters
        vertices (np.ndarray): [N, 3] 3-dimensional vertices
        faces (np.ndarray): [T, 3] triangular face indices

    ## Returns
        vertices (np.ndarray): [N_, 3] 3-dimensional vertices
        faces (np.ndarray): [Q, 4] quad face indices
    """
    raise NotImplementedError
