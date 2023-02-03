""" STAR data format to ICG data format
We need to create a tracker folder for ICG
Required files:
- detector.yaml
- reconstruct.obj
"""

import os
import numpy as np
import yaml
import json
import open3d as o3d
import cv2

def np_mat_to_yaml(mat):
    yaml_mat = {}
    yaml_mat["rows"] = mat.shape[0]
    yaml_mat["cols"] = mat.shape[1]
    yaml_mat["dt"] = "d"
    yaml_mat["data"] = mat.flatten().tolist()
    return yaml_mat


def generate_icg_tracker(tracker_name, data_dir, icg_dir):
    # generate the obj file if not exist
    obj_path = os.path.join(data_dir, "star", "reconstruct.obj")
    if not os.path.exists(obj_path):
        print("Generating the obj file...")
        pcd_path = os.path.join(data_dir, "star", "reconstruct.pcd")
        pcd = o3d.io.read_point_cloud(pcd_path)
        # transfer point cloud to obj using possion
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        o3d.io.write_triangle_mesh(obj_path, mesh)
    
    # copy the obj file
    icg_obj_path = os.path.join(icg_dir, "reconstruct.obj")
    os.makedirs(os.path.dirname(icg_obj_path), exist_ok=True)
    os.system("cp {} {}".format(obj_path, icg_obj_path))

    # save the config yaml
    config_yaml_path = os.path.join(icg_dir, "config.yaml")
    config_s = cv2.FileStorage(config_yaml_path, cv2.FileStorage_WRITE)

    # save the camera yaml
    config_s.startWriteStruct("LoaderColorCamera", cv2.FileNode_SEQ)
    config_s.startWriteStruct("", cv2.FileNode_MAP)
    config_s.write("name", "loader_color")
    config_s.write("metafile_path", "camera_color.yaml")
    config_s.endWriteStruct()
    config_s.endWriteStruct()

    # save the normal color viewer
    config_s.startWriteStruct("NormalColorViewer", cv2.FileNode_SEQ)
    config_s.startWriteStruct("", cv2.FileNode_MAP)
    config_s.write("name", "color_viewer")
    config_s.write("color_camera", "loader_color")
    config_s.write("renderer_geometry", "renderer_geometry")
    config_s.endWriteStruct()
    config_s.endWriteStruct()

    # save the renderer geometry
    config_s.startWriteStruct("RendererGeometry", cv2.FileNode_SEQ)
    config_s.startWriteStruct("", cv2.FileNode_MAP)
    config_s.write("name", "renderer_geometry")
    config_s.write("bodies", [tracker_name])
    config_s.endWriteStruct()
    config_s.endWriteStruct()

    # save the body
    config_s.startWriteStruct("Body", cv2.FileNode_SEQ)
    config_s.startWriteStruct("", cv2.FileNode_MAP)
    config_s.write("name", tracker_name)
    config_s.write("metafile_path", "object.yaml")
    config_s.endWriteStruct()
    config_s.endWriteStruct()

    # save the static detector
    config_s.startWriteStruct("StaticDetector", cv2.FileNode_SEQ)
    config_s.startWriteStruct("", cv2.FileNode_MAP)
    config_s.write("name", "static_detector")
    config_s.write("metafile_path", "detector.yaml")
    config_s.write("body", tracker_name)
    config_s.write("color_camera", "loader_color")
    config_s.endWriteStruct()
    config_s.endWriteStruct()

    # save the region model
    config_s.startWriteStruct("RegionModel", cv2.FileNode_SEQ)
    config_s.startWriteStruct("", cv2.FileNode_MAP)
    config_s.write("name", "region_model")
    config_s.write("metafile_path", "model.yaml")
    config_s.write("body", tracker_name)
    config_s.endWriteStruct()
    config_s.endWriteStruct()

    # save the region modality
    config_s.startWriteStruct("RegionModality", cv2.FileNode_SEQ)
    config_s.startWriteStruct("", cv2.FileNode_MAP)
    config_s.write("name", "region_modality")
    config_s.write("body", tracker_name)
    config_s.write("color_camera", "loader_color")
    config_s.write("region_model", "region_model")
    config_s.endWriteStruct()
    config_s.endWriteStruct()

    # save the optimizer
    config_s.startWriteStruct("Optimizer", cv2.FileNode_SEQ)
    config_s.startWriteStruct("", cv2.FileNode_MAP)
    config_s.write("name", "optimizer")
    config_s.write("modalities", "region_modality")
    config_s.endWriteStruct()
    config_s.endWriteStruct()
    
    # save the tracker
    config_s.startWriteStruct("Tracker", cv2.FileNode_SEQ)
    config_s.startWriteStruct("", cv2.FileNode_MAP)
    config_s.write("name", "tracker")
    config_s.write("viewers", ["color_viewer"])
    config_s.write("detectors", ["static_detector"])
    config_s.write("optimizers", ["optimizer"])
    config_s.endWriteStruct()
    config_s.endWriteStruct()
    config_s.release()

    # save the cam color
    context_json_file = os.path.join(data_dir, "star", "context.json")
    with open(context_json_file, "r") as f:
        context_data = json.load(f)

    config_yaml_path = os.path.join(icg_dir, "camera_color.yaml")
    cam_color_s = cv2.FileStorage(config_yaml_path, cv2.FileStorage_WRITE)
    cam_color_s.write("load_directory", os.path.join(data_dir, "color"))
    cam_color_s.startWriteStruct("intrinsics", cv2.FileNode_MAP)
    cam_color_s.write("f_u", context_data["cam-00"]["intrinsic"][0])
    cam_color_s.write("f_v", context_data["cam-00"]["intrinsic"][1])
    cam_color_s.write("pp_x", context_data["cam-00"]["intrinsic"][2])
    cam_color_s.write("pp_v", context_data["cam-00"]["intrinsic"][3])
    cam_color_s.write("width", context_data["cam-00"]["image_cols"])
    cam_color_s.write("height", context_data["cam-00"]["image_rows"])
    cam_color_s.endWriteStruct()
    extrinsic = np.array(context_data["cam-00"]["extrinsic"]).reshape(4, 4)
    cam_color_s.write("camara2world_pose", extrinsic)
    cam_color_s.write("depth_scale", 1.0)
    cam_color_s.write("image_name_pre", "")
    cam_color_s.write("load_index", 0)
    cam_color_s.write("n_leading_zeros", 0)
    cam_color_s.write("image_name_post", "")
    cam_color_s.write("load_image_type", "png")
    cam_color_s.release()

    # save the detector
    kf_results = np.load(os.path.join(args.data_dir, "star", "kf_results.npz"))
    cam_poses = kf_results["cam_poses"]
    config_yaml_path = os.path.join(icg_dir, "detector.yaml")
    detector_s = cv2.FileStorage(config_yaml_path, cv2.FileStorage_WRITE)
    detector_s.write("body2world_pose", cam_poses[0])
    detector_s.release()

    # save the model
    config_yaml_path = os.path.join(icg_dir, "model.yaml")
    model_s = cv2.FileStorage(config_yaml_path, cv2.FileStorage_WRITE)
    model_s.write("model_path", "INFER_FROM_NAME")
    model_s.release()

    # save the object
    config_yaml_path = os.path.join(icg_dir, "object.yaml")
    object_s = cv2.FileStorage(config_yaml_path, cv2.FileStorage_WRITE)
    object_s.write("geometry_path", "reconstruct.obj")
    object_s.write("geometry_unit_in_meter", 1.0)
    object_s.write("geometry_counterclockwise", 1)
    object_s.write("geometry_enable_culling", 1)
    object_s.write("geometry2body_pose", np.eye(4))
    object_s.release()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker_name", type=str, default="", help="name")
    parser.add_argument("--data_dir", type=str, default="", help="src data folder")
    parser.add_argument("--icg_dir", type=str, default="", help="exported icg data folder")
    args = parser.parse_args()
    # generate the icg tracker config
    generate_icg_tracker(args.tracker_name, args.data_dir, args.icg_dir)