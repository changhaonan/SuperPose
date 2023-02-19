import open3d as o3d
import glob
import os
import yaml
import cv2
import numpy as np

def show_cad_model(model_dir, sfm_dir=None):
    # read the init pose
    detector_yaml_file = os.path.join(model_dir, "detector.yaml")
    # parse the yaml file using opencv
    detector_yaml = cv2.FileStorage(detector_yaml_file, cv2.FILE_STORAGE_READ)
    # read the init pose
    init_pose = detector_yaml.getNode("body2world_pose").mat()
    cad_file = glob.glob(os.path.join(model_dir, "*.obj"))[0]
    # read the sfm poses from sfm_dir
    sfm_poses = dict()
    if sfm_dir is not None:
        for sfm_file in glob.glob(os.path.join(sfm_dir, "*.txt")):
            np.loadtxt(sfm_file)
            pose_id = os.path.basename(sfm_file).split(".")[0].split("_")[-1]
            sfm_poses[pose_id] = np.loadtxt(sfm_file)
    sfm_poses_vis = []
    for pose_id in sfm_poses:
        sfm_pose = sfm_poses[pose_id]
        # sfm_pose = np.linalg.inv(sfm_pose)
        sfm_pose = np.matmul(init_pose, sfm_pose)
        sfm_pose_frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.1)
        sfm_pose_frame.transform(sfm_pose)
        sfm_poses_vis.append(sfm_pose_frame)

    # read the mesh using open3d with texture
    cad_model = o3d.io.read_triangle_mesh(cad_file, True)
    # scale the model by 1000 at origin
    cad_model.scale(1.0/1000.0, center=np.zeros(3, dtype=np.float64))
    # apply the init pose
    cad_model.transform(init_pose)
    obj_frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.1)
    obj_frame.transform(init_pose)
    # draw origin
    origin_frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.1)
    # o3d.visualization.draw_geometries([cad_model, origin_frame])
    # set camera pose
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(cad_model)
    vis.add_geometry(origin_frame)
    vis.add_geometry(obj_frame)
    for sfm_pose_vis in sfm_poses_vis:
        vis.add_geometry(sfm_pose_vis)
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, -1, 0])
    ctr.set_zoom(0.5)
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="", help="path to model")
    parser.add_argument("--sfm_dir", type=str, default="", help="path to sfm")
    args = parser.parse_args()

    # visualize cad model
    show_cad_model(args.model_dir, args.sfm_dir)
