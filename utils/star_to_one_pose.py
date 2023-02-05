""" STAR data format to OnePose data format
STAR data: 
- images: star/cam-00/frame-%06d.color.png (color image)
- images: star/cam-00/frame-%06d.depth.png (depth image)
- intrinsics: star/context.json (camera intrinsics)
- poses: star/kf_results.npz (camera poses)
- reconstruct: star/reconstruct.pcd (point cloud)
OnePose data:
- color images: color/%d.png
- depth images: depth/%d.png
- poses: poses_ba/%d.txt
- intrinsics: intrinsics.txt
"""

import os
import open3d as o3d
import numpy as np
import json


def bbox_o3d_to_onepose(bbox_o3d):
    # order
    o3d_order = [0, 1, 2, 3, 4, 5, 6, 7]
    onepose_order = [0, 4, 3, 1, 6, 2, 5, 7]
    bbox_onepose = np.zeros((8, 3))
    for i, j in zip(onepose_order, o3d_order):
        bbox_onepose[i, :] = bbox_o3d[j, :]
    return bbox_onepose


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="", help="data folder")
    args = parser.parse_args()
    
    print("Start converting STAR data to colmap data...")

    # generating bbox
    pcd = o3d.io.read_point_cloud(os.path.join(args.data_dir, "star", "reconstruct.pcd"))
    bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(pcd.points)
    bbox.color = (0, 0, 1)
    bbox_corners = bbox.get_box_points()
    bbox_corners = bbox_o3d_to_onepose(np.asarray(bbox_corners))  # Convert to onepose format
    bbox_path = os.path.join(args.data_dir, "../", "box3d_corners.txt")
    np.savetxt(bbox_path, bbox_corners)

    pcd_frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.2)
    # save the bbox to "../corner3d.txt"
    # o3d.visualization.draw_geometries([pcd, bbox, pcd_frame])

    # transfer the pose & intrinsics
    kf_results = np.load(os.path.join(args.data_dir, "star", "kf_results.npz"))
    star_poses = kf_results["cam_poses"]
    context_json_file = os.path.join(args.data_dir, "star", "context.json")
    with open(context_json_file, "r") as f:
        context_data = json.load(f)
    star_intr = np.array(
        [
            [context_data["cam-00"]["intrinsic"][0], 0, context_data["cam-00"]["intrinsic"][2]],
            [0, context_data["cam-00"]["intrinsic"][1], context_data["cam-00"]["intrinsic"][3]],
            [0, 0, 1]
        ]
    )
    pose_dir = os.path.join(args.data_dir, "poses_ba")
    intr_dir = os.path.join(args.data_dir, "intrin_ba")
    intr_file = os.path.join(args.data_dir, "intrinsics.txt")
    os.makedirs(pose_dir, exist_ok=True)
    os.makedirs(intr_dir, exist_ok=True)
    
    for i, star_pose in enumerate(star_poses):
        # extrinsic
        colmap_pose = np.linalg.inv(star_pose)  # colmap pose = inv(star pose)
        with open(os.path.join(pose_dir, "{}.txt".format(i)), "w") as f:
            f.writelines(
                [
                    "{} {} {} {}\n".format(*colmap_pose[0, :]),
                    "{} {} {} {}\n".format(*colmap_pose[1, :]),
                    "{} {} {} {}\n".format(*colmap_pose[2, :]),
                    "{} {} {} {}\n".format(*colmap_pose[3, :])
                ],
            )
        # intrinsic
        with open(os.path.join(intr_dir, "{}.txt".format(i)), "w") as f:
            f.writelines(
                [
                    "{} {} {}\n".format(*star_intr[0, :]),
                    "{} {} {}\n".format(*star_intr[1, :]),
                    "{} {} {}\n".format(*star_intr[2, :])
                ],
            )
    with open(intr_file, "w") as f:
        f.writelines(
            [
                "fx: {}\n".format(context_data["cam-00"]["intrinsic"][0]),
                "fy: {}\n".format(context_data["cam-00"]["intrinsic"][1]),
                "cx: {}\n".format(context_data["cam-00"]["intrinsic"][2]),
                "cy: {}\n".format(context_data["cam-00"]["intrinsic"][3]),
            ],
        )
    print("Finish converting STAR data to colmap data...")