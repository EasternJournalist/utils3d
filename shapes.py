import numpy as np

def cube():
    vertices = np.array([
        [-0.5, 0.5, 0.5],   [0.5, 0.5, 0.5],   [0.5, -0.5, 0.5],   [-0.5, -0.5, 0.5], # v0-v1-v2-v3
        [-0.5, 0.5, -0.5],  [0.5, 0.5, -0.5],  [0.5, -0.5, -0.5],  [-0.5, -0.5, -0.5] # v4-v5-v6-v7
    ], dtype=np.float32).reshape((-1, 3))

    faces = np.array([
        [0, 1, 2, 3], # v0-v1-v2-v3 (front)
        [4, 5, 1, 0], # v4-v5-v1-v0 (top)
        [3, 2, 6, 7], # v3-v2-v6-v7 (bottom)
        [5, 4, 7, 6], # v5-v4-v7-v6 (back)
        [1, 5, 6, 2], # v1-v5-v6-v2 (right)
        [4, 0, 3, 7]  # v4-v0-v3-v7 (left)
    ], dtype=int)

    return vertices, faces
