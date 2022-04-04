import numpy as np

def to_linear_depth(depth_buffer: np.ndarray, near: float, far: float):
    return (2 * near * far) / (far + near - (2 * depth_buffer - 1) * (far - near))

def triangulate(indices: np.ndarray):
    assert len(indices.shape) == 2
    if indices.shape[1] == 3:
        return indices
    n = indices.shape[1]
    loop_indice = np.stack([np.zeros(n - 2, dtype=int), np.arange(1, n - 1, 1, dtype=int), np.arange(2, n, 1, dtype=int)], axis=1)
    return indices[:, loop_indice].reshape(-1, 3)

def perspective_from_image(fov: float, width: float, height: float, near: float, far: float):
    return np.array([
        [1. / (np.tan(fov / 2) * (width / max(width, height))), 0., 0., 0.],
        [0., 1. / (np.tan(fov / 2) * (height / max(width, height))), 0., 0.],
        [0., 0., (near + far) / (near - far), 2. * near * far / (near - far)],
        [0., 0., -1., 0.] 
    ])

def perspective_from_fov_xy(fov_x: float, fov_y: float, near: float, far: float):
    return np.array([
        [1. / np.tan(fov_x / 2), 0., 0., 0.],
        [0., 1. / np.tan(fov_y / 2), 0., 0.],
        [0., 0., (near + far) / (near - far), 2. * near * far / (near - far)],
        [0., 0., -1., 0.] 
    ])

def image_uv(width: int, height: int) -> np.ndarray:
    u = np.linspace(0.5 / width, 1. - 0.5 / width, width)
    v = np.linspace(0.5 / height, 1. - 0.5 / height, height)
    return np.concatenate([u[None, :, None].repeat(height, axis=0), v[:, None, None].repeat(width, axis=1)], axis=2)