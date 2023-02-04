""" 
Shift the geometry model to origin, and the camera poses.
"""

import numpy as np
import cv2
import os
import open3d as o3d


def shift_model(data_dir):
    # load the point cloud
    pcd_path = os.path.join(data_dir, "star", "reconstruct.pcd")
    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd_points = np.asarray(pcd.points)
    shift = np.mean(pcd_points, axis=0)
    
    if np.linalg.norm(shift) > 1e-6:
        # shift the point cloud
        pcd.translate(-shift)
        o3d.io.write_point_cloud(pcd_path, pcd)
        # shift the camera poses
        kf_results = np.load(os.path.join(data_dir, "star", "kf_results.npz"))
        cam_poses = kf_results["cam_poses"]
        for i in range(len(cam_poses)):
            cam_poses[i][:3, 3] -= shift
        np.savez(os.path.join(data_dir, "star", "kf_results.npz"), cam_poses=cam_poses)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="", help="path to model")
    args = parser.parse_args()

    shift_model(args.data_dir)