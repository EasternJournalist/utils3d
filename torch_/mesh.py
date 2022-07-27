import torch


def triangulate(faces: torch.Tensor) -> torch.Tensor:
    assert len(faces.shape) == 2
    if faces.shape[1] == 3:
        return faces
    n = faces.shape[1]
    loop_indice = torch.stack([
        torch.zeros(n - 2, dtype=torch.int64), 
        torch.arange(1, n - 1, 1, dtype=torch.int64), 
        torch.arange(2, n, 1, dtype=torch.int64)
    ], dim=1)
    return faces[:, loop_indice].reshape(-1, 3)

def compute_face_normal(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Compute face normals of a triangular mesh

    Args:
        vertices (np.ndarray):  3-dimensional vertices of shape (..., N, 3)
        faces (np.ndarray): triangular face indices of shape (T, 3)

    Returns:
        normals (np.ndarray): face normals of shape (..., T, 3)
    """
    normal = torch.cross(torch.index_select(vertices, dim=-2, index=faces[:, 1]) - torch.index_select(vertices, dim=-2, index=faces[:, 0]), torch.index_select(vertices, dim=-2, index=faces[:, 2]) - torch.index_select(vertices, dim=-2, index=faces[:, 0]))
    normal = normal / (torch.norm(normal, p=2, dim=-1, keepdim=True) + 1e-7)
    return normal

def compute_vertex_normal(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Compute vertex normals as mean of adjacent face normals

    Args:
        vertices (np.ndarray): 3-dimensional vertices of shape (..., N, 3)
        faces (np.ndarray): triangular face indices of shape (T, 3)

    Returns:
        normals (np.ndarray): vertex normals of shape (..., N, 3)
    """
    face_normal = compute_face_normal(vertices, faces) # (..., T, 3)
    face_normal = face_normal[..., None, :].repeat(*[1] * (len(vertices.shape) - 1), 3, 1).view(*face_normal.shape[:-2], -1, 3) # (..., T * 3, 3)
    vertex_normal = torch.index_add(torch.zeros_like(vertices), dim=-2, index=faces.view(-1), source=face_normal)
    vertex_normal = vertex_normal / (torch.norm(vertex_normal, p=2, dim=-1, keepdim=True) + 1e-7)
    return vertex_normal

def compute_face_tbn(pos: torch.Tensor, faces_pos: torch.Tensor, uv: torch.Tensor, faces_uv: torch.Tensor) -> torch.Tensor:
    """compute TBN matrix for each face

    Args:
        pos (torch.Tensor): shape (..., N_pos, 3), positions
        faces_pos (torch.Tensor): shape(T, 3) 
        uv (torch.Tensor): shape (..., N_uv, 3) uv coordinates, 
        faces_uv (torch.Tensor): shape(T, 3) 
        
    Returns:
        torch.Tensor: (..., T, 3, 3) TBN matrix for each face. Note TBN vectors is normalized but not orthognal
    """
    e01 = torch.index_select(pos, dim=-2, index=faces_pos[:, 1]) - torch.index_select(pos, dim=-2, index=faces_pos[:, 0])
    e02 = torch.index_select(pos, dim=-2, index=faces_pos[:, 2]) - torch.index_select(pos, dim=-2, index=faces_pos[:, 0])
    uv01 = torch.index_select(uv, dim=-2, index=faces_uv[:, 1]) - torch.index_select(uv, dim=-2, index=faces_uv[:, 0])
    uv02 = torch.index_select(uv, dim=-2, index=faces_uv[:, 2]) - torch.index_select(uv, dim=-2, index=faces_uv[:, 0])
    normal = torch.cross(e01, e02)
    tangent_bitangent = torch.stack([e01, e02], dim=-1) @ torch.inverse(torch.stack([uv01, uv02], dim=-1))
    tbn = torch.cat([tangent_bitangent, normal.unsqueeze(-1)], dim=-1)
    tbn = tbn / (torch.norm(tbn, p=2, dim=-1, keepdim=True) + 1e-7)
    return tbn

def compute_vertex_tbn(faces_topo: torch.Tensor, pos: torch.Tensor, faces_pos: torch.Tensor, uv: torch.Tensor, faces_uv: torch.Tensor) -> torch.Tensor:
    """compute TBN matrix for each face

    Args:
        faces_topo (torch.Tensor): (..., T, 3), face indice of topology
        pos (torch.Tensor): shape (..., N_pos, 3), positions
        faces_pos (torch.Tensor): shape(T, 3) 
        uv (torch.Tensor): shape (..., N_uv, 3) uv coordinates, 
        faces_uv (torch.Tensor): shape(T, 3) 
        
    Returns:
        torch.Tensor: (..., T, 3, 3) TBN matrix for each face. Note TBN vectors is normalized but not orthognal
    """
    n_vertices = faces_topo.max().item() + 1
    n_tri = faces_topo.shape[-2]
    batch_shape = faces_topo.shape[:-2]
    face_tbn = compute_face_tbn(pos, faces_pos, uv, faces_uv)    # (..., T, 3, 3)
    face_tbn = face_tbn[..., :, None, :, :].repeat(*[1] * len(batch_shape), 1, 3, 1, 1).view(*batch_shape, n_tri * 3, 3, 3)   # (..., T * 3, 3, 3)
    vertex_tbn = torch.index_add(torch.zeros(*batch_shape, n_vertices, 3, 3).to(face_tbn), dim=-3, index=faces_topo.view(-1), source=face_tbn)
    vertex_tbn = vertex_tbn / (torch.norm(vertex_tbn, p=2, dim=-1, keepdim=True) + 1e-7)
    return vertex_tbn

def laplacian_smooth_mesh(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Laplacian smooth with cotangent weights

    Args:
        vertices (torch.Tensor): shape (..., N, 3)
        faces (torch.Tensor): shape (T, 3)
    """
    sum_verts = torch.zeros_like(vertices)                          # (..., N, 3)
    sum_weights = torch.zeros(*vertices.shape[:-1]).to(vertices)    # (..., N)
    face_verts = torch.index_select(vertices, -2, faces.view(-1)).view(*vertices.shape[:-2], *faces.shape, 3)   # (..., T, 3)
    for i in range(3):
        e1 = face_verts[..., (i + 1) % 3, :] - face_verts[..., i, :]
        e2 = face_verts[..., (i + 2) % 3, :] - face_verts[..., i, :]
        cos_angle = (e1 * e2).sum(dim=-1) / (e1.norm(p=2, dim=-1) * e2.norm(p=2, dim=-1))
        cot_angle = cos_angle / (1 - cos_angle ** 2) ** 0.5         # (..., T, 3)
        sum_verts = torch.index_add(sum_verts, -2, faces[:, (i + 1) % 3], face_verts[..., (i + 2) % 3, :] * cot_angle[..., None])
        sum_weights = torch.index_add(sum_weights, -1, faces[:, (i + 1) % 3], cot_angle)
        sum_verts = torch.index_add(sum_verts, -2, faces[:, (i + 2) % 3], face_verts[..., (i + 1) % 3, :] * cot_angle[..., None])
        sum_weights = torch.index_add(sum_weights, -1, faces[:, (i + 2) % 3], cot_angle)
    return sum_verts / (sum_weights[..., None] + 1e-7)

def laplacian_smooth_mesh(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Laplacian smooth with cotangent weights

    Args:
        vertices (torch.Tensor): shape (..., N, 3)
        faces (torch.Tensor): shape (T, 3)
    """
    sum_verts = torch.zeros_like(vertices)                          # (..., N, 3)
    sum_weights = torch.zeros(*vertices.shape[:-1]).to(vertices)    # (..., N)
    face_verts = torch.index_select(vertices, -2, faces.view(-1)).view(*vertices.shape[:-2], *faces.shape, 3)   # (..., T, 3)
    for i in range(3):
        e1 = face_verts[..., (i + 1) % 3, :] - face_verts[..., i, :]
        e2 = face_verts[..., (i + 2) % 3, :] - face_verts[..., i, :]
        cos_angle = (e1 * e2).sum(dim=-1) / (e1.norm(p=2, dim=-1) * e2.norm(p=2, dim=-1))
        cot_angle = cos_angle / (1 - cos_angle ** 2) ** 0.5         # (..., T, 3)
        sum_verts = torch.index_add(sum_verts, -2, faces[:, (i + 1) % 3], face_verts[..., (i + 2) % 3, :] * cot_angle[..., None])
        sum_weights = torch.index_add(sum_weights, -1, faces[:, (i + 1) % 3], cot_angle)
        sum_verts = torch.index_add(sum_verts, -2, faces[:, (i + 2) % 3], face_verts[..., (i + 1) % 3, :] * cot_angle[..., None])
        sum_weights = torch.index_add(sum_weights, -1, faces[:, (i + 2) % 3], cot_angle)
    return sum_verts / (sum_weights[..., None] + 1e-7)

def taubin_smooth_mesh(vertices: torch.Tensor, faces: torch.Tensor, lambda_: float = 0.5, mu_: float = -0.51) -> torch.Tensor:
    """Taubin smooth mesh

    Args:
        vertices (torch.Tensor): _description_
        faces (torch.Tensor): _description_
        lambda_ (float, optional): _description_. Defaults to 0.5.
        mu_ (float, optional): _description_. Defaults to -0.51.

    Returns:
        torch.Tensor: _description_
    """
    pt = vertices + lambda_ * laplacian_smooth_mesh(vertices, faces)
    p = pt + mu_ * laplacian_smooth_mesh(pt, faces)
    return p
