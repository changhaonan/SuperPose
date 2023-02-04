import open3d as o3d
import numpy as np
import os


def vis_pcd(pcd, cam_poses=None, coord_frame_size=0.2):
    vis_list = []
    # add pcd and frame
    if len(pcd) == 1:
        # create bbox from pcd
        bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(pcd.points)
        bbox.color = (0, 0, 1)
        pcd_frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=coord_frame_size)
        vis_list = [*pcd, pcd_frame, bbox]
    else:
        vis_list = [*pcd]
    
    # add camera poses
    if cam_poses is not None:
        for cam_pose in cam_poses:
            cam_frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=coord_frame_size)
            cam_frame.transform(cam_pose)
            vis_list.append(cam_frame)
        o3d.visualization.draw_geometries(vis_list)
    else:
        o3d.visualization.draw_geometries(vis_list)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="", help="path to model")
    args = parser.parse_args()

    # prepare
    kf_results = np.load(os.path.join(args.model_path, "kf_results.npz"))
    cam_poses = kf_results["cam_poses"]
    for key in kf_results.keys():
        print(key)
    # Debug the renderer
    test_pose_1 = np.array(
        [
            [0, 0, 1, -1.6],
            [0, 1, 0, -0.8,],
            [-1, 0, 0, 0,],
            [0, 0, 0, 1,],
        ]
    )
    test_pose_2 = cam_poses[-1]
    print(np.linalg.inv(test_pose_2))
    # trans origin
    origin = np.array([[0, 0, 0, 1]])
    print(np.matmul(test_pose_2, origin.T))

    # read the point cloud using open3d
    pcd = o3d.io.read_point_cloud(os.path.join(args.model_path, "reconstruct.pcd"))
    
    # create a unit sphere
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)

    # visualization
    vis_pcd([pcd, sphere], [test_pose_1, test_pose_2])