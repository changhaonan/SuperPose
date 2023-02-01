import open3d as o3d 
import numpy as np
import os

def vis_pcd(pcd, cam_poses=None, coord_frame_size=0.2):
    # create bbox from pcd
    bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(pcd.points)
    bbox.color = (0, 0, 1)
    pcd_frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=coord_frame_size)
    if cam_poses is not None:
        vis_list = [pcd, pcd_frame, bbox]
        for cam_pose in cam_poses:
            cam_frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=coord_frame_size)
            cam_frame.transform(cam_pose)
            vis_list.append(cam_frame)
        o3d.visualization.draw_geometries(vis_list)
    else:
        o3d.visualization.draw_geometries([bbox, pcd, pcd_frame])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="", help="path to model")
    args = parser.parse_args()

    # read the point cloud using open3d
    pcd = o3d.io.read_point_cloud(os.path.join(args.model_path, "cracker_box.pcd"))
    # read the camera pose
    kf_results = np.load(os.path.join(args.model_path, "kf_results.npz"))
    # vis them
    vis_pcd(pcd, kf_results["cam_poses"])