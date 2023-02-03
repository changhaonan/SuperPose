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

def np_mat_to_yaml(mat):
    yaml_mat = {}
    yaml_mat["rows"] = mat.shape[0]
    yaml_mat["cols"] = mat.shape[1]
    yaml_mat["dt"] = "d"
    yaml_mat["data"] = mat.flatten().tolist()
    return yaml_mat


def generate_icg_tracker(tracker_name, data_dir, icg_dir):
    # copy the obj file
    obj_path = os.path.join(data_dir, "star", "reconstruct.obj")
    icg_obj_path = os.path.join(icg_dir, "reconstruct.obj")
    os.makedirs(os.path.dirname(icg_obj_path), exist_ok=True)
    os.system("cp {} {}".format(obj_path, icg_obj_path))
    # prepare the camera color yaml
    context_json_file = os.path.join(data_dir, "star", "context.json")
    with open(context_json_file, "r") as f:
        context_data = json.load(f)
    
    # prepare the camera color yaml
    cam_color_yaml = {}
    cam_color_yaml["load_directory"] = os.path.join(data_dir, "color")
    cam_color_yaml["intrinsics"] = {}
    cam_color_yaml["intrinsics"]["f_u"] = context_data["cam-00"]["intrinsic"][0]
    cam_color_yaml["intrinsics"]["f_v"] = context_data["cam-00"]["intrinsic"][1]
    cam_color_yaml["intrinsics"]["pp_x"] = context_data["cam-00"]["intrinsic"][2]
    cam_color_yaml["intrinsics"]["pp_y"] = context_data["cam-00"]["intrinsic"][3]
    cam_color_yaml["intrinsics"]["width"] = context_data["cam-00"]["image_cols"]
    cam_color_yaml["intrinsics"]["height"] = context_data["cam-00"]["image_rows"]
    cam_color_yaml["camera2world_pose"] = np_mat_to_yaml(context_data["cam-00"]["extrinsic"])
    cam_color_yaml["depth_scale"] = 1.0
    cam_color_yaml["image_name_pre"] = ""
    cam_color_yaml["load_index"] = 0
    cam_color_yaml["n_leading_zeros"] = 0
    cam_color_yaml["image_name_post"] = ""
    cam_color_yaml["load_image_type"] = "png"
    # save the yaml
    cam_color_yaml_path = os.path.join(icg_dir, "camera_color.yaml")
    with open(cam_color_yaml_path, "w") as f:
        yaml.dump(cam_color_yaml, f)

    # prepare the detector yaml (Manual detector set the pose in the first frame)
    kf_results = np.load(os.path.join(args.data_dir, "star", "kf_results.npz"))
    cam_poses = kf_results["cam_poses"]
    detector_yaml = {}
    detector_yaml["body2world_pose"] = np_mat_to_yaml(cam_poses[0])
    # save the yaml
    detector_yaml_path = os.path.join(icg_dir, "detector.yaml")
    with open(detector_yaml_path, "w") as f:
        yaml.dump(detector_yaml, f)

    # prepare the object yaml
    obj_yaml = {}
    obj_yaml["geometry_path"] = "reconstruct.obj"
    obj_yaml["geometry_unit_in_meters"] = 1.0  # scale
    obj_yaml["geometry_counterclockwise"] = 1
    obj_yaml["geometry_enable_culling"] = 1
    obj_yaml["geometry2body_pose"] = np_mat_to_yaml(np.eye(4))
    # save the yaml
    obj_yaml_path = os.path.join(icg_dir, "object.yaml")
    with open(obj_yaml_path, "w") as f:
        yaml.dump(obj_yaml, f)
    
    # default model yaml
    model_yaml = {}
    model_yaml["model_path"] = "INFER_FROM_NAME"
    # save the yaml
    model_yaml_path = os.path.join(icg_dir, "model.yaml")
    with open(model_yaml_path, "w") as f:
        yaml.dump(model_yaml, f)
    
    # finally, the config yaml
    config_yaml = {}
    config_yaml["LoaderColorCamera"] = {}
    config_yaml["LoaderColorCamera"]["name"] = "loader_color"
    config_yaml["LoaderColorCamera"]["metafile_path"] = "camera_color.yaml"

    config_yaml["NormalColorViewer"] = {}
    config_yaml["NormalColorViewer"]["name"] = "color_viewer"
    config_yaml["NormalColorViewer"]["color_camera"] = "loader_color"
    config_yaml["NormalColorViewer"]["renderer_geometry"] = "renderer_geometry"

    config_yaml["RendererGeometry"] = {}
    config_yaml["RendererGeometry"]["name"] = "renderer_geometry"
    config_yaml["RendererGeometry"]["bodies"] = [tracker_name]

    config_yaml["Body"] = {}
    config_yaml["Body"]["name"] = tracker_name
    config_yaml["Body"]["metafile_path"] = "object.yaml"

    config_yaml["StaticDetector"] = {}
    config_yaml["StaticDetector"]["name"] = "static_detector"
    config_yaml["StaticDetector"]["metafile_path"] = "detector.yaml"
    config_yaml["body"] = tracker_name
    config_yaml["color_camera"] = "loader_color"

    config_yaml["RegionModel"] = {}
    config_yaml["RegionModel"]["name"] = "region_model"
    config_yaml["RegionModel"]["metafile_path"] = "model.yaml"
    config_yaml["body"] = tracker_name

    config_yaml["RegionModality"] = {}
    config_yaml["RegionModality"]["name"] = "region_modality"
    config_yaml["RegionModality"]["body"] = tracker_name
    config_yaml["RegionModality"]["color_camera"] = "loader_color"
    config_yaml["RegionModality"]["region_model"] = "region_model"

    config_yaml["Optimizer"] = {}
    config_yaml["Optimizer"]["name"] = "optimizer"
    config_yaml["Optimizer"]["modalities"] = "region_modality"

    config_yaml["Tracker"] = {}
    config_yaml["Tracker"]["name"] = "tracker"
    config_yaml["Tracker"]["viewers"] = ["color_viewer"]
    config_yaml["Tracker"]["detectors"] = ["static_detector"]
    config_yaml["Tracker"]["optimizers"] = ["optimizer"]

    # save the yaml
    config_yaml_path = os.path.join(icg_dir, "config.yaml")
    with open(config_yaml_path, "w") as f:
        yaml.dump(config_yaml, f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker_name", type=str, default="", help="name")
    parser.add_argument("--data_dir", type=str, default="", help="src data folder")
    parser.add_argument("--icg_dir", type=str, default="", help="exported icg data folder")
    args = parser.parse_args()
    # generate the icg tracker config
    generate_icg_tracker(args.tracker_name, args.data_dir, args.icg_dir)