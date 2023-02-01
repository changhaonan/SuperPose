""" STAR data format to colmap data format
STAR data: 
- images: star/cam-00/frame-%06d.color.png (color image)
- images: star/cam-00/frame-%06d.depth.png (depth image)
- intrinsics: star/context.json (camera intrinsics)
- poses: star/kf_results.npz (camera poses)
- reconstruct: star/reconstruct.pcd (point cloud)
Colmap data:
- color images: color/%d.png
- depth images: depth/%d.png
- poses: poses_ba/%d.txt
- intrinsics: intrinsics.txt
"""

import os
import numpy as np

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="", help="data folder")
    args = parser.parse_args()
    
    print("Start converting STAR data to colmap data...")
    # transfer the pose & intrinsics
    kf_results = np.load(os.path.join(args.data_dir, "star", "kf_results.npz"))
    star_poses = kf_results["cam_poses"]
    star_intr = kf_results["cam_intr"]
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
                "fx: {}\n".format(star_intr[0, 0]),
                "fy: {}\n".format(star_intr[1, 1]),
                "cx: {}\n".format(star_intr[0, 2]),
                "cy: {}\n".format(star_intr[1, 2]),
            ],
        )
    print("Finish converting STAR data to colmap data...")