import numpy as np
import open3d as o3d

# Transfer colmap Points3D to open3d PointCloud:
def points_colmap_to_o3d(points3D):
    points3D_list = []
    for k, v in points3D.items():
        points3D_list.append(v.xyz)
    points_o3d = o3d.geometry.PointCloud()
    points_o3d.points = o3d.utility.Vector3dVector(np.stack(points3D_list, axis=0))
    return points_o3d


def bbox_o3d_to_onepose(bbox_o3d):
    # order
    o3d_order = [0, 1, 2, 3, 4, 5, 6, 7]
    onepose_order = [0, 4, 3, 1, 6, 2, 5, 7]
    bbox_onepose = np.zeros((8, 3))
    for i, j in zip(onepose_order, o3d_order):
        bbox_onepose[i, :] = bbox_o3d[j, :]
    return bbox_onepose