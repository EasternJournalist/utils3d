from utils3d.helpers import timeit
import utils3d
from tqdm import trange
import numpy as np
import time

import open3d as o3d
import trimesh
import plyfile
import meshio

test_file = "debug/dragon_poly.ply"


def u3d_read_ply(file_path):
    data = utils3d.np.read_ply(file_path)
    data['vertex']['x'] += 0  # dummy operation to prevent optimization
    return data

def plyfile_read_ply(file_path):
    plydata = plyfile.PlyData.read(file_path)
    vertex = plydata['vertex']
    x = vertex['x']  # dummy operation to prevent optimization
    x += 0
    return plydata

def trimesh_read_ply(file_path):
    mesh = trimesh.load_mesh(file_path, preprocess=False)
    return mesh

def o3d_read_ply(file_path):
    # mesh = o3d.io.read_triangle_mesh(file_path)
    # np.asanyarray(mesh.vertices)
    # np.asanyarray(mesh.triangles)
    # return mesh
    pcd = o3d.io.read_point_cloud(file_path)
    np.asarray(pcd.points)
    return pcd

def meshio_read_ply(file_path):
    mesh = meshio.read(file_path)
    return mesh


reader = {
    'utils3d': u3d_read_ply,
    # 'Open3D': o3d_read_ply,
    # 'Trimesh': trimesh_read_ply,
    # 'plyfile': plyfile_read_ply,
    # 'meshio': meshio_read_ply,
}


def plyfile_write_ply(file_path, data):
    data.write(file_path)

def trimesh_write_ply(file_path, data):
    data.export(file_path)

def o3d_write_ply(file_path, data):
    # o3d.io.write_triangle_mesh(file_path, data)
    o3d.io.write_point_cloud(file_path, data)

def meshio_write_ply(file_path, data):
    meshio.write(file_path, data)


writer = {
    'utils3d': utils3d.np.write_ply,
    'Open3D': o3d_write_ply,
    'Trimesh': trimesh_write_ply,
    'plyfile': plyfile_write_ply,
    'meshio': meshio_write_ply,
}

reader_times = []
writer_times = []
ROUND = 10
for name, func in reader.items():
    try:
        data = func(test_file)  # warm up
    except Exception as e:
        print(f"Error while testing {name}: {e}")
        reader_times.append(float('inf'))
        writer_times.append(float('inf'))
        continue
    start_read_time = time.time()
    for _ in trange(ROUND, desc=f"Testing {name}"):
        data = func(test_file)
    end_read_time = time.time()
    reader_times.append((end_read_time - start_read_time) / ROUND)

    start_write_time = time.time()
    for _ in trange(ROUND, desc=f"Testing {name} write"):
        writer[name](f"debug/{name}_out.ply", data)
    end_write_time = time.time()
    average_write_time = (end_write_time - start_write_time) / ROUND
    writer_times.append(average_write_time)

# print row in markdown table format
print(" | ".join(reader.keys()))
print(" | ".join([f"{t * 1000:.1f} ms" for t in reader_times]))
print(" | ".join(writer.keys()))
print(" | ".join([f"{t * 1000:.1f} ms" for t in writer_times]))
