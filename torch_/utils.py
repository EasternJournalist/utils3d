import torch
from typing import Tuple, Union
from numbers import Number

from ..numpy_.utils import (
    perspective_from_fov as __perspective_from_fov, 
    perspective_from_fov_xy as __perspective_from_fov_xy,
    image_uv as __image_uv,
    image_mesh as __image_mesh,
    to_linear_depth as __to_linear_depth,
    to_depth_buffer as __to_depth_buffer,
    chessboard as __chessboard
)

def to_linear_depth(depth_buffer: torch.Tensor) -> torch.Tensor:
    return __to_linear_depth(depth_buffer)

def to_depth_buffer(linear_depth: torch.Tensor) -> torch.Tensor:
    return __to_depth_buffer(linear_depth)

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

def rodrigues(rot_vecs: torch.Tensor) -> torch.Tensor:
    """Calculates the rotation matrices for a batch of rotation vectors. (code from SMPLX)

    Args:
        rot_vecs (torch.Tensor): shape (..., 3), axis-angle vetors

    Returns:
        torch.Tensor: shape (..., 3, 3) The rotation matrices for the given axis-angle parameters
    """
    batch_shape = rot_vecs.shape[:-1]
    device, dtype = rot_vecs.device, rot_vecs.dtype

    angle = torch.norm(rot_vecs + 1e-8, dim=-1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.cos(angle)[..., None, :]
    sin = torch.sin(angle)[..., None, :]

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=-1)
    K = torch.zeros((*batch_shape, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((*batch_shape, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1).view((*batch_shape, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat

def perspective_from_image(fov: float, width: int, height: int, near: float, far: float) -> torch.Tensor:
    return torch.from_numpy(__perspective_from_fov(fov, width, height, near, far))

def perspective_from_fov_xy(fov_x: float, fov_y: float, near: float, far: float) -> torch.Tensor:
    return torch.from_numpy(__perspective_from_fov_xy(fov_x, fov_y, near, far))

def perspective_to_intrinsic(perspective: torch.Tensor) -> torch.Tensor:
    """OpenGL convention perspective matrix to OpenCV convention intrinsic

    Args:
        perspective (torch.Tensor): shape (4, 4) or (..., 4, 4), OpenGL convention perspective matrix

    Returns:
        torch.Tensor: shape (3, 3) or (..., 3, 3), OpenCV convention intrinsic
    """
    fx, fy = perspective[..., 0, 0], perspective[..., 1, 1]
    cx, cy = perspective[..., 0, 2], perspective[..., 1, 2]
    zero = torch.zeros_like(fx)
    one = torch.full_like(fx, -1)

    matrix = [
        [0.5 * fx,     zero, -0.5 * cx + 0.5],
        [    zero, 0.5 * fy,  0.5 * cy + 0.5],
        [    zero,     zero,             one]]
    return torch.stack([torch.stack(row, dim=-1) for row in matrix], dim=-2)

def intrinsic_to_perspective(intrinsic: torch.Tensor, near: float, far: float) -> torch.Tensor:
    """OpenGL convention perspective matrix to OpenCV convention intrinsic

    Args:
        intrinsic (torch.Tensor): shape (3, 3) or (..., 3, 3), OpenCV convention intrinsic
        near (float): near plane to clip
        far (float): far plane to clip
    Returns:
        torch.Tensor: shape (4, 4) or (..., 4, 4), OpenGL convention perspective matrix
    """
    fx, fy = intrinsic[..., 0, 0], intrinsic[..., 1, 1]
    cx, cy = intrinsic[..., 0, 2], intrinsic[..., 1, 2]
    zero = torch.zeros_like(fx)
    negone = torch.full_like(fx, -1)
    a = torch.full_like(fx, (near + far) / (near - far))
    b = torch.full_like(fx, 2. * near * far / (near - far))

    matrix = [
        [2 * fx,   zero,  -2 * cx + 1, zero],
        [  zero, 2 * fy,   2 * cy - 1, zero],
        [  zero,   zero,            a,    b],
        [  zero,   zero,       negone, zero]]
    return torch.stack([torch.stack(row, dim=-1) for row in matrix], dim=-2)

def extrinsic_to_view(extrinsic: torch.Tensor) -> torch.Tensor:
    """OpenCV convention camera extrinsic to OpenGL convention view matrix

    Args:
        extrinsic (torch.Tensor): shape (4, 4) or (..., 4, 4), OpenCV convention camera extrinsic

    Returns:
        torch.Tensor: shape (4, 4) or (..., 4, 4) OpenGL convention view matrix
    """
    return torch.inverse(extrinsic) @ torch.diag(torch.tensor([1, -1, -1, 1])).to(extrinsic)

def view_to_extrinsic(view: torch.Tensor) -> torch.Tensor:
    """OpenCV convention camera extrinsic to OpenGL convention view matrix

    Args:
        view (torch.Tensor): shape (4, 4) or (..., 4, 4), OpenGL convention view matrix

    Returns:
        torch.Tensor: shape (4, 4) or (..., 4, 4) OpenCV convention camera extrinsic
    """
    return torch.diag(torch.tensor([1, -1, -1, 1])).to(view) @ torch.inverse(view)

def camera_cv_to_gl(extrinsic: torch.Tensor, intrinsic: torch.Tensor, near: float, far: float):
    """Convert OpenCV convention camera extrinsic & intrinsic to OpenGL convention view matrix and perspective matrix

    Args:
        extrinsic (torch.Tensor): shape (4, 4) or (..., 4, 4), OpenCV convention camera extrinsic
        intrinsic (torch.Tensor): shape (3, 3) or (..., 3, 3), OpenCV convention intrinsic
        near (float): near plane to clip
        far (float): far plane to clip

    Returns:
        view (torch.Tensor): shape (4, 4) or (..., 4, 4), OpenGL convention view matrix
        perspective (torch.Tensor): shape (4, 4) or (..., 4, 4), OpenGL convention perspective matrix
    """
    return extrinsic_to_view(extrinsic), intrinsic_to_perspective(intrinsic, near, far)

def camera_gl_to_cv(view, perspective):
    """Convert OpenGL convention view matrix & perspective matrix to OpenCV convention camera extrinsic & intrinsic 

    Args:
        view (torch.Tensor): shape (4, 4) or (..., 4, 4), OpenGL convention view matrix
        perspective (torch.Tensor): shape (4, 4) or (..., 4, 4), OpenGL convention perspective matrix

    Returns:
        view (torch.Tensor): shape (4, 4) or (..., 4, 4), OpenCV convention camera extrinsic
        perspective (torch.Tensor): shape (3, 3) or (..., 3, 3), OpenCV convention intrinsic
    """
    return view_to_extrinsic(view), perspective_to_intrinsic(perspective)

def normalize_intrinsic(intrinsic: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """normalize camera intrinsic
    Args:
        intrinsic (torch.Tensor): shape (..., 3, 3) 
        width (int): image width
        height (int): image height

    Returns:
        (torch.Tensor): shape (..., 3, 3), same as input intrinsic. Normalized intrinsic(s)
    """
    return intrinsic * torch.tensor([1 / width, 1 / height, 1])[:, None].to(intrinsic)

def crop_intrinsic(intrinsic: torch.Tensor, width: int, height: int, left: int, top: int, crop_width: int, crop_height: int):
    """Evaluate the new intrinsic(s) after crop the image: cropped_img = img[top:bottom, left:right]

    Args:
        intrinsic (torch.Tensor): shape (3, 3), a normalized camera intrinsic
        width (int): 
        height (int): 
        top (int): 
        left (int): 
        bottom (int): 
        right (int): 
    """
    s = torch.tensor([
        [width / crop_width, 0, -left / crop_width], 
        [0, height / crop_height,  -top / crop_height], 
        [0., 0., 1.]]).to(intrinsic)
    return s @ intrinsic

def image_uv(width: int, height: int):
    return torch.from_numpy(__image_uv(width, height))

def image_mesh(width: int, height: int, mask: torch.Tensor = None):
    uv, faces = __image_mesh(width, height, mask.cpu().numpy() if mask is not None else None)
    uv, faces = torch.from_numpy(uv), torch.from_numpy(faces)
    if mask is not None:
        uv, faces= uv.to(mask.device), faces.to(mask.device)
    return uv, faces

def projection(vertices: torch.Tensor, model_matrix: torch.Tensor = None, view_matrix: torch.Tensor = None, projection_matrix: torch.Tensor = None) -> torch.Tensor:
    """Project 3D points to 2D following the OpenGL convention (except for row major matrice)

    Args:
        vertices (torch.Tensor): 3D vertices positions of shape (batch, n, 3)
        model_matrix (torch.Tensor): row major model to world matrix of shape (batch, 4, 4)
        view_matrix (torch.Tensor): camera to world matrix of shape (batch, 4, 4)
        projection_matrix (torch.Tensor): projection matrix of shape (batch, 4, 4)

    Returns:
        scr_coord (torch.Tensor): vertex screen space coordinates of shape (batch, n, 3), value ranging in [0, 1]. 
            first two channels are screen space uv, where the origin (0., 0.) is corresponding to the bottom-left corner of the screen.
        linear_depth (torch.Tensor): 
    """
    if model_matrix is None: model_matrix = torch.eye(4, dtype=vertices.dtype).to(vertices)[None, ...]
    if view_matrix is None: view_matrix = torch.eye(4, dtype=vertices.dtype).to(vertices)[None, ...]
    if projection_matrix is None: projection_matrix = torch.eye(4, dtype=vertices.dtype).to(vertices)[None, ...]

    vertices = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
    clip_coord = vertices @ (projection_matrix @ torch.inverse(view_matrix) @ model_matrix).transpose(dim0=-2, dim1=-1)
    ndc_coord = clip_coord[..., :3] / clip_coord[..., 3:]
    scr_coord = ndc_coord * 0.5 + 0.5
    linear_depth = clip_coord[..., 3]
    return scr_coord, linear_depth

def projection_ndc(vertices: torch.Tensor, model_matrix: torch.Tensor = None, view_matrix: torch.Tensor = None, projection_matrix: torch.Tensor = None) -> torch.Tensor:
    """Very similar to projection(), but return NDC space coordinates instead of screen space coordinates.
    The only difference is scr_coord = ndc_coord * 0.5 + 0.5.

    Args:
        vertices (torch.Tensor): 3D vertices positions of shape (batch, n, 3)
        model_matrix (torch.Tensor): row major model to world matrix of shape (batch, 4, 4)
        view_matrix (torch.Tensor): camera to world matrix of shape (batch, 4, 4)
        projection_matrix (torch.Tensor): projection matrix of shape (batch, 4, 4)

    Returns:
        ndc_coord (torch.Tensor): NDC space coordinates of shape (batch, n, 3), value ranging in [-1, 1]. Point (-1, -1, -1) is corresponding to the left & bottom & nearest.
    """
    if model_matrix is None: model_matrix = torch.eye(4, dtype=vertices.dtype).to(vertices)[None, ...]
    if view_matrix is None: view_matrix = torch.eye(4, dtype=vertices.dtype).to(vertices)[None, ...]
    if projection_matrix is None: projection_matrix = torch.eye(4, dtype=vertices.dtype).to(vertices)[None, ...]

    vertices = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
    clip_coord = vertices @ (projection_matrix @ torch.inverse(view_matrix) @ model_matrix).transpose(dim0=-2, dim1=-1)
    ndc_coord = clip_coord[..., :3] / clip_coord[..., 3:]
    linear_depth = clip_coord[..., 3]
    return ndc_coord, linear_depth

def chessboard(width: int, height: int, grid_size: int, color_a: torch.Tensor, color_b: torch.Tensor) -> torch.Tensor:
    """get a chessboard image

    Args:
        width (int): image width
        height (int): image height
        grid_size (int): size of chessboard grid
        color_a (torch.Tensor): shape (chanenls,), color of the grid at the top-left corner
        color_b (torch.Tensor): shape (chanenls,), color in complementary grids

    Returns:
        image (np.ndarray): shape (height, width, channels), chessboard image
    """
    x = torch.div(torch.arange(width), grid_size, rounding_mode='floor')
    y = torch.div(torch.arange(height), grid_size, rounding_mode='floor')
    mask = ((x[None, :] + y[:, None]) % 2).to(color_a)
    image = (1 - mask[..., None]) * color_a + mask[..., None] * color_b
    return image