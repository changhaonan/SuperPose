"""
Transfer CAD model to RGB-Recon
"""
import open3d as o3d
import numpy as np
import glob
import os
import cv2
import tqdm
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
    # fix the seed
    np.random.seed(0)
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
    # prepare the sfm
    color_dir = os.path.join(sfm_path, "color")
    poses_dir = os.path.join(sfm_path, "poses_ba")
    intrin_dir = os.path.join(sfm_path, "intrin_ba")
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(poses_dir, exist_ok=True)
    os.makedirs(intrin_dir, exist_ok=True)
    # clean
    for f in glob.glob(os.path.join(color_dir, "*.png")):
        os.remove(f)
    for f in glob.glob(os.path.join(poses_dir, "*.txt")):
        os.remove(f)
    for f in glob.glob(os.path.join(intrin_dir, "*.txt")):
        os.remove(f)
    # change the pose and render
    K = 200  # Number of camera poses to generate
    radius = 0.2  # Radius of the sphere
    camera_poses = generate_camera_poses(K, radius)
    camera_poses = sort_camera_poses_by_rotation(camera_poses)
    # set up the intrinsic camera parameters
    width, height = 640, 480
    focal = 500  # Focal lengths
    cx, cy = width / 2 - 0.5, height / 2 - 0.5  # Principal point
    intrinsic_params = o3d.camera.PinholeCameraIntrinsic(width, height, focal, focal, cx, cy)
    camera_params = o3d.camera.PinholeCameraParameters()
    camera_params.intrinsic = intrinsic_params
    pose_list = camera_poses
    for i, pose in tqdm.tqdm(enumerate(pose_list)):
        new_pose = np.linalg.inv(pose)
        # flip the z axis
        new_pose[2, :] *= -1
        camera_params.extrinsic = new_pose
        # set the scene with the point cloud
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=width, height=height, visible=False)
        vis.add_geometry(cad_model)
        # visualize coordinate
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        frame.transform(pose)
        vis.add_geometry(frame)
        vis.get_view_control().convert_from_pinhole_camera_parameters(camera_params)
        # capture the image and close the visualizer
        image = vis.capture_screen_float_buffer(do_render=True)
        vis.destroy_window()
        # convert the image to a numpy array
        image = np.asarray(image)
        # convert the image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # save the images
        image_name = os.path.join(color_dir, f"{i}.png")
        cv2.imwrite(image_name, image * 255)
        # save the pose
        output_pose = np.linalg.inv(pose)
        output_pose[1, :] *= -1
        output_pose[2, :] *= -1
        np.savetxt(os.path.join(poses_dir, f"{i}.txt"), output_pose)
        # save the intrinsic matrix
        np.savetxt(os.path.join(intrin_dir, f"{i}.txt"), intrinsic_params.intrinsic_matrix)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cad_path", type=str, default="data/RBOT_dataset/bakingsoda")
    parser.add_argument("--sfm_path", type=str, default="data/RBOT_dataset/bakingsoda/sfm")
    parser.add_argument("--color_dir", type=str, default="data/sfm_model/bakingsoda")
    args = parser.parse_args()
    cad2video(args.cad_path, args.color_dir)
