"""
Transfer CAD model to RGB-Recon
"""
import open3d as o3d
import numpy as np
import glob
import os
import cv2

import numpy as np
from scipy.spatial.transform import Rotation as R


def fibonacci_sphere_sampling(K, radius=1):
    indices = np.arange(0, K, dtype=float) + 0.5

    phi = np.arccos(1 - 2 * indices / K)
    theta = np.pi * (1 + 5**0.5) * indices

    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)

    points = np.array([x, y, z]).T
    return radius * points


def generate_camera_poses(K, radius=1):
    # Sample K points on the sphere
    points = fibonacci_sphere_sampling(K, radius)

    # Generate the camera poses
    camera_poses = []
    for point in points:
        z_axis = -point / np.linalg.norm(point)
        x_axis = np.cross(z_axis, np.array([0, 1, 0]))
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        rotation_matrix = np.stack([x_axis, y_axis, z_axis], axis=1)
        translation = -point
        # translation = np.array([0, 0, 0])

        camera_pose = np.eye(4)
        camera_pose[:3, :3] = rotation_matrix
        camera_pose[:3, 3] = translation
        camera_poses.append(camera_pose)

    return np.array(camera_poses)


def rotation_distance(pose1, pose2):
    R1 = pose1[:3, :3]
    R2 = pose2[:3, :3]
    R_diff = np.dot(R1, R2.T)
    trace = np.trace(R_diff)
    angle = np.arccos((trace - 1) / 2)
    return angle


def sort_camera_poses_by_rotation(camera_poses):
    sorted_poses = []
    unprocessed_poses = list(camera_poses)

    # Randomly pick the first camera pose
    first_pose_idx = np.random.randint(len(unprocessed_poses))
    current_pose = unprocessed_poses.pop(first_pose_idx)
    sorted_poses.append(current_pose)

    while unprocessed_poses:
        min_distance = float("inf")
        closest_pose_idx = None

        # Find the closest pose in rotation to the current one
        for i, pose in enumerate(unprocessed_poses):
            distance = rotation_distance(current_pose, pose)
            if distance < min_distance:
                min_distance = distance
                closest_pose_idx = i

        # Update the current pose and add it to the sorted list
        current_pose = unprocessed_poses.pop(closest_pose_idx)
        sorted_poses.append(current_pose)

    return np.array(sorted_poses)


def cad2video(cad_path, sfm_path):
    """Transfer CAD to video"""
    # read the mesh using open3d with texture
    cad_file = glob.glob(os.path.join(cad_path, "*.obj"))[0]
    cad_model = o3d.io.read_triangle_mesh(cad_file, True)
    # scale the model by 1000 at origin
    cad_model.scale(1.0 / 1000.0, center=np.zeros(3, dtype=np.float64))

    init_pose = np.eye(4)
    # apply the init pose
    cad_model.transform(init_pose)

    # visualize
    # o3d.visualization.draw_geometries([cad_model])

    # prepare the sfm
    output_dir = os.path.join(sfm_path, "images")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # change the pose and render
    K = 200  # Number of camera poses to generate
    radius = 0.2  # Radius of the sphere
    camera_poses = generate_camera_poses(K, radius)
    camera_poses = sort_camera_poses_by_rotation(camera_poses)

    prev_pose = np.eye(4)
    pose_list = camera_poses
    for i, pose in enumerate(pose_list):
        # new_pose = np.linalg.inv(pose)
        new_pose = pose
        pose_delta = new_pose @ np.linalg.inv(prev_pose)
        prev_pose = pose
        cad_model.transform(pose_delta)
        width, height = 400, 400

        # Set the scene with the point cloud
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=width, height=height, visible=False)
        vis.add_geometry(cad_model)
        vis.update_renderer()
        # Capture the image and close the visualizer
        image = vis.capture_screen_float_buffer(do_render=True)
        vis.destroy_window()

        # Convert the image to a numpy array
        image = np.asarray(image)
        # Convert the image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Save the images
        image_name = os.path.join(output_dir, f"{i:04d}.png")
        cv2.imwrite(image_name, image * 255)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cad_path", type=str, default="data/RBOT_dataset/bakingsoda")
    parser.add_argument("--sfm_path", type=str, default="data/RBOT_dataset/bakingsoda/sfm")
    parser.add_argument("--output_dir", type=str, default="data/sfm_model/bakingsoda")
    args = parser.parse_args()
    cad2video(args.cad_path, args.output_dir)
