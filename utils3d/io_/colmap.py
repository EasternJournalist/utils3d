from typing import *
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


__all__ = ['read_extrinsics_from_colmap', 'read_intrinsics_from_colmap', 'write_extrinsics_as_colmap', 'write_intrinsics_as_colmap']


def write_extrinsics_as_colmap(file: Union[str, Path], extrinsics: np.ndarray, image_names: Union[str, List[str]] = 'image_{i:04d}.png', camera_ids: List[int] = None):
    assert extrinsics.shape[1:] == (4, 4) and extrinsics.ndim == 3 or extrinsics.shape == (4, 4)
    if extrinsics.ndim == 2:
        extrinsics = extrinsics[np.newaxis, ...]
    quats = Rotation.from_matrix(extrinsics[:, :3, :3]).as_quat()
    trans = extrinsics[:, :3, 3]
    if camera_ids is None:
        camera_ids = list(range(1, len(extrinsics) + 1))
    if isinstance(image_names, str):
        image_names = [image_names.format(i=i) for i in range(1, len(extrinsics) + 1)]
    assert len(extrinsics) == len(image_names) == len(camera_ids), \
        f'Number of extrinsics ({len(extrinsics)}), image_names ({len(image_names)}), and camera_ids ({len(camera_ids)}) must be the same'
    with open(file, 'w') as fp:
        print("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME", file=fp)
        for i, (quat, t, name, camera_id) in enumerate(zip(quats.tolist(), trans.tolist(), image_names, camera_ids)):
            # Colmap has wxyz order while scipy.spatial.transform.Rotation has xyzw order. Haha, wcnm.
            qx, qy, qz, qw = quat
            tx, ty, tz = t
            print(f'{i + 1} {qw:f} {qx:f} {qy:f} {qz:f} {tx:f} {ty:f} {tz:f} {camera_id:d} {name}', file=fp)
            print()


def write_intrinsics_as_colmap(file: Union[str, Path], intrinsics: np.ndarray, width: int, height: int, normalized: bool = False):
    assert intrinsics.shape[1:] == (3, 3) and intrinsics.ndim == 3 or intrinsics.shape == (3, 3)
    if intrinsics.ndim == 2:
        intrinsics = intrinsics[np.newaxis, ...]
    if normalized:
        intrinsics = intrinsics * np.array([width, height, 1])[:, None]
    with open(file, 'w') as fp:
        print("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]", file=fp)
        for i, intr in enumerate(intrinsics):
            fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
            print(f'{i + 1} PINHOLE {width:d} {height:d} {fx:f} {fy:f} {cx:f} {cy:f}', file=fp)


def read_extrinsics_from_colmap(file: Union[str, Path]) -> Union[np.ndarray, List[int], List[str]]:
    with open(file) as fp:
        lines = fp.readlines()
    image_names, quats, trans, camera_ids = [], [], [], []
    i_line = 0
    for line in lines:
        line = line.strip()
        if line.startswith('#'):
            continue
        i_line += 1
        if i_line % 2 == 0:
            continue
        image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name = line.split()
        quats.append([float(qx), float(qy), float(qz), float(qw)])
        trans.append([float(tx), float(ty), float(tz)])
        camera_ids.append(int(camera_id))
        image_names.append(name)
    
    quats = np.array(quats, dtype=np.float32)
    trans = np.array(trans, dtype=np.float32)
    rotation = Rotation.from_quat(quats).as_matrix()
    extrinsics = np.concatenate([
        np.concatenate([rotation, trans[..., None]], axis=-1), 
        np.array([0, 0, 0, 1], dtype=np.float32)[None, None, :].repeat(len(quats), axis=0)
    ], axis=-2)

    return extrinsics, camera_ids, image_names


def read_intrinsics_from_colmap(file: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    with open(file) as fp:
        lines = fp.readlines()
    intrinsics, distortions = [], []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        camera_id, model, width, height, *params = line.split()
        if model == 'PINHOLE':
            fx, fy, cx, cy = map(float, params[:4])
            k1 = k2 = k3 = p1 = p2 = 0.0
        elif model == 'OPENCV':
            fx, fy, cx, cy, k1, k2, p1, p2, k3 = *map(float, params[:8]), 0.0
        elif model == 'SIMPLE_RADIAL':
            f, cx, cy, k = map(float, params[:4])
            fx = fy = f
            k1, k2, p1, p2, k3 = k, 0.0, 0.0, 0.0, 0.0
        intrinsics.append([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        distortions.append([k1, k2, p1, p2, k3])
    intrinsics = np.array(intrinsics, dtype=np.float32)
    distortions = np.array(distortions, dtype=np.float32)
    return intrinsics, distortions