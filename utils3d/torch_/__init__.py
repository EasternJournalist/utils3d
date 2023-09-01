import warnings
import importlib

from . import utils
from . import transforms
from . import mesh


from .utils import (    
    image_uv,
    image_mesh,
    chessboard
)

from .transforms import (
    perspective,
    perspective_from_fov,
    perspective_from_fov_xy,
    intrinsic,
    intrinsic_from_fov,
    intrinsic_from_fov_xy,
    perspective_to_intrinsic,
    intrinsic_to_perspective,
    view_look_at,
    extrinsic_look_at,
    extrinsic_to_view,
    view_to_extrinsic,
    normalize_intrinsic,
    crop_intrinsic,
    pixel_to_uv,
    pixel_to_ndc,
    project_depth,
    linearize_depth,
    project_gl,
    project_cv,
    unproject_gl,
    unproject_cv,

    euler_angles_to_matrix,
    rodrigues,
)

from .mesh import (
    triangulate,
    compute_face_normal,
    compute_face_angle,
    compute_vertex_normal,
    compute_vertex_normal_weighted,
    remove_corrupted_faces,
    merge_duplicate_vertices,
    subdivide_mesh_simple,
    
    laplacian,
    compute_face_tbn,
    compute_vertex_tbn,
    laplacian_smooth_mesh,
    taubin_smooth_mesh,
    laplacian_hc_smooth_mesh,
)

if importlib.find_loader('nvdiffrast'):
    from .rasterization import *
else:
    warnings.warn('nvdiffrast not found, rasterization functions are not available.')