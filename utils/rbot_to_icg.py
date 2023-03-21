""" RBOT data format to ICG generator data format
"""
import os
import numpy as np
import yaml
import json
import open3d as o3d
import cv2


def generate_icg_tracker(
    tracker_name,
    model_dir,
    eval_dir,
    icg_dir,
    use_network_detector=False,
    detector_port=8080,
    feature_port=9090,
    enable_depth=False,
    enable_feature=False,
    enable_refine=False,
):
    # config macro
    OBJECT_SCALE = 2.0  # the size of object, influencing the accept threshold for depth modality

    os.makedirs(icg_dir, exist_ok=True)
    os.system(
        "cp {} {}".format(
            os.path.join(model_dir, f"{tracker_name}.obj"),
            os.path.join(icg_dir, f"{tracker_name}.obj"),
        )
    )
    os.system(
        "cp {} {}".format(
            os.path.join(model_dir, f"{tracker_name}_tex.png"),
            os.path.join(icg_dir, f"{tracker_name}_tex.png"),
        )
    )
    os.system(
        "cp {} {}".format(
            os.path.join(model_dir, f"{tracker_name}.obj.mtl"),
            os.path.join(icg_dir, f"{tracker_name}.obj.mtl"),
        )
    )

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

    # save the network detector
    if use_network_detector:
        detector_method = "NetworkDetector"
    else:
        detector_method = "StaticDetector"
    config_s.startWriteStruct(detector_method, cv2.FileNode_SEQ)
    config_s.startWriteStruct("", cv2.FileNode_MAP)
    config_s.write("name", "detector")
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
    config_s.write("metafile_path", "region_modality.yaml")
    if enable_depth:
        config_s.startWriteStruct("measure_occlusions", cv2.FileNode_MAP)
        config_s.write("depth_camera", "loader_depth")
        config_s.endWriteStruct()
    config_s.endWriteStruct()
    config_s.endWriteStruct()

    if enable_depth:
        # save the depth camera
        config_s.startWriteStruct("LoaderDepthCamera", cv2.FileNode_SEQ)
        config_s.startWriteStruct("", cv2.FileNode_MAP)
        config_s.write("name", "loader_depth")
        config_s.write("metafile_path", "camera_depth.yaml")
        config_s.endWriteStruct()
        config_s.endWriteStruct()

        # save the depth viewer
        config_s.startWriteStruct("NormalDepthViewer", cv2.FileNode_SEQ)
        config_s.startWriteStruct("", cv2.FileNode_MAP)
        config_s.write("name", "depth_viewer")
        config_s.write("depth_camera", "loader_depth")
        config_s.write("renderer_geometry", "renderer_geometry")
        config_s.endWriteStruct()
        config_s.endWriteStruct()

        # save the depth model
        config_s.startWriteStruct("DepthModel", cv2.FileNode_SEQ)
        config_s.startWriteStruct("", cv2.FileNode_MAP)
        config_s.write("name", "depth_model")
        config_s.write("metafile_path", "model.yaml")
        config_s.write("body", tracker_name)
        config_s.endWriteStruct()
        config_s.endWriteStruct()

        # save the depth modality
        config_s.startWriteStruct("DepthModality", cv2.FileNode_SEQ)
        config_s.startWriteStruct("", cv2.FileNode_MAP)
        config_s.write("name", "depth_modality")
        config_s.write("body", tracker_name)
        config_s.write("depth_camera", "loader_depth")
        config_s.write("depth_model", "depth_model")
        config_s.write("metafile_path", "depth_modality.yaml")
        config_s.endWriteStruct()
        config_s.endWriteStruct()

    if enable_feature:
        # save the feature viewer
        config_s.startWriteStruct("FeatureViewer", cv2.FileNode_SEQ)
        config_s.startWriteStruct("", cv2.FileNode_MAP)
        config_s.write("name", "feature_viewer")
        config_s.write("color_camera", "loader_color")
        config_s.write("renderer_geometry", "renderer_geometry")
        config_s.write("metafile_path", "feature_viewer.yaml")
        config_s.endWriteStruct()
        config_s.endWriteStruct()

        # save the feature model
        config_s.startWriteStruct("FeatureModel", cv2.FileNode_SEQ)
        config_s.startWriteStruct("", cv2.FileNode_MAP)
        config_s.write("name", "feature_model")
        config_s.write("metafile_path", "feature_model.yaml")
        config_s.write("body", tracker_name)
        config_s.endWriteStruct()
        config_s.endWriteStruct()

        # save the feature modality
        config_s.startWriteStruct("FeatureModality", cv2.FileNode_SEQ)
        config_s.startWriteStruct("", cv2.FileNode_MAP)
        config_s.write("name", "feature_modality")
        config_s.write("body", tracker_name)
        config_s.write("color_camera", "loader_color")
        if enable_depth:
            config_s.write("depth_camera", "loader_depth")
        config_s.write("feature_model", "feature_model")
        config_s.write("metafile_path", "feature_modality.yaml")
        config_s.endWriteStruct()
        config_s.endWriteStruct()

    # save the optimizer
    config_s.startWriteStruct("Optimizer", cv2.FileNode_SEQ)
    config_s.startWriteStruct("", cv2.FileNode_MAP)
    config_s.write("name", "optimizer")
    modality_list = ["region_modality"]
    if enable_depth:
        modality_list.append("depth_modality")
    if enable_feature and not enable_refine:
        modality_list.append("feature_modality")
    config_s.write("metafile_path", "optimizer.yaml")
    config_s.write("modalities", modality_list)
    config_s.endWriteStruct()
    if enable_refine and enable_feature:
        config_s.startWriteStruct("", cv2.FileNode_MAP)
        config_s.write("name", "feature_optimizer")
        modality_list = ["feature_modality"]
        config_s.write("metafile_path", "optimizer.yaml")
        config_s.write("modalities", modality_list)
        config_s.endWriteStruct()
    config_s.endWriteStruct()

    # save the refiner
    if enable_refine and enable_feature:
        config_s.startWriteStruct("Refiner", cv2.FileNode_SEQ)
        config_s.startWriteStruct("", cv2.FileNode_MAP)
        config_s.write("name", "feature_refiner")
        config_s.write("optimizers", ["feature_optimizer"])
        config_s.write("metafile_path", "refiner.yaml")
        config_s.endWriteStruct()
        config_s.endWriteStruct()

    # save the tracker
    config_s.startWriteStruct("Tracker", cv2.FileNode_SEQ)
    config_s.startWriteStruct("", cv2.FileNode_MAP)
    config_s.write("name", "tracker")
    viewer_list = ["color_viewer"]
    if enable_depth:
        viewer_list.append("depth_viewer")
    if enable_feature:
        viewer_list.append("feature_viewer")
    config_s.write("viewers", viewer_list)
    config_s.write("detectors", ["detector"])
    config_s.write("optimizers", ["optimizer"])
    config_s.write("refiners", ["feature_refiner"])
    config_s.write("metafile_path", "tracker.yaml")
    config_s.endWriteStruct()
    config_s.endWriteStruct()

    config_s.release()

    # save the cam color
    # load camera info
    camera_config_file = os.path.join(model_dir, "../camera_calibration.txt")
    with open(camera_config_file, "r") as f:
        camera_data = f.readlines()
    fx = float(camera_data[1].split("\t")[0])
    fy = float(camera_data[1].split("\t")[1])
    cx = float(camera_data[1].split("\t")[2])
    cy = float(camera_data[1].split("\t")[3])
    config_yaml_path = os.path.join(icg_dir, "camera_color.yaml")
    cam_color_s = cv2.FileStorage(config_yaml_path, cv2.FileStorage_WRITE)
    cam_color_s.write("load_directory", os.path.join(eval_dir, "frames"))
    cam_color_s.startWriteStruct("intrinsics", cv2.FileNode_MAP)
    cam_color_s.write("f_u", fx)
    cam_color_s.write("f_v", fy)
    cam_color_s.write("pp_x", cx)
    cam_color_s.write("pp_y", cy)
    cam_color_s.write("width", 640)
    cam_color_s.write("height", 512)
    cam_color_s.endWriteStruct()
    cam_color_s.write("camara2world_pose", np.eye(4))
    cam_color_s.write("depth_scale", 1.0)
    cam_color_s.write("image_name_pre", "a_regular")
    cam_color_s.write("load_index", 0)
    cam_color_s.write("n_leading_zeros", 4)
    cam_color_s.write("image_name_post", "")
    cam_color_s.write("load_image_type", "png")
    cam_color_s.release()

    # save the cam depth
    config_yaml_path = os.path.join(icg_dir, "camera_depth.yaml")
    cam_depth_s = cv2.FileStorage(config_yaml_path, cv2.FileStorage_WRITE)
    cam_depth_s.write("load_directory", os.path.join(eval_dir, "frames"))
    cam_depth_s.startWriteStruct("intrinsics", cv2.FileNode_MAP)
    cam_depth_s.write("f_u", fx)
    cam_depth_s.write("f_v", fy)
    cam_depth_s.write("pp_x", cx)
    cam_depth_s.write("pp_y", cy)
    cam_depth_s.write("width", 640)
    cam_depth_s.write("height", 512)
    cam_depth_s.endWriteStruct()
    cam_depth_s.write("camara2world_pose", np.eye(4))
    cam_depth_s.write("depth_scale", 1.0)
    cam_depth_s.write("image_name_pre", "a_regular")
    cam_depth_s.write("load_index", 0)
    cam_depth_s.write("n_leading_zeros", 4)
    cam_depth_s.write("image_name_post", "")
    cam_depth_s.write("load_image_type", "png")
    cam_depth_s.release()

    # save the detector
    # load the init pose
    pose_file = os.path.join(eval_dir, "../poses_first.txt")
    with open(pose_file, "r") as f:
        pose_data = f.readlines()
    init_pose_list = [float(x) for x in pose_data[1].split("\t")]
    init_pose = np.array(
        [
            init_pose_list[0:3] + [init_pose_list[9] * 0.001],
            init_pose_list[3:6] + [init_pose_list[10] * 0.001],
            init_pose_list[6:9] + [init_pose_list[11] * 0.001],
            [0, 0, 0, 1],
        ]
    )
    init_pose = np.linalg.inv(init_pose)
    config_yaml_path = os.path.join(icg_dir, "detector.yaml")
    detector_s = cv2.FileStorage(config_yaml_path, cv2.FileStorage_WRITE)
    detector_s.write("body2world_pose", np.linalg.inv(init_pose))  # the object init position
    if use_network_detector:
        detector_s.write("port", detector_port)
        detector_s.write("reinit_iter", 10)  # reinit the detector every 10 frames
    else:
        detector_s.write("reinit_iter", 0)  # no reinit
    detector_s.release()

    # save the model
    config_yaml_path = os.path.join(icg_dir, "model.yaml")
    model_s = cv2.FileStorage(config_yaml_path, cv2.FileStorage_WRITE)
    model_s.write("model_path", "INFER_FROM_NAME")
    model_s.release()

    # save the feature model
    config_yaml_path = os.path.join(icg_dir, "feature_model.yaml")
    model_s = cv2.FileStorage(config_yaml_path, cv2.FileStorage_WRITE)
    model_s.write("model_path", "INFER_FROM_NAME")
    model_s.write("n_features", 200)
    model_s.release()

    # save the modality file
    config_yaml_path = os.path.join(icg_dir, "region_modality.yaml")
    modality_s = cv2.FileStorage(config_yaml_path, cv2.FileStorage_WRITE)
    # rbot-related param
    modality_s.write("n_lines", 200)
    modality_s.write("min_continuous_distance", 6.0)
    modality_s.write("function_length", 8)
    modality_s.write("distribution_length", 12)
    modality_s.write("function_amplitude", 0.36)
    modality_s.write("function_slope", 0.0)
    modality_s.write("learning_rate", 1.3)
    modality_s.startWriteStruct("scales", cv2.FileNode_SEQ)
    modality_s.write("", 5)
    modality_s.write("", 2)
    modality_s.write("", 2)
    modality_s.write("", 1)
    modality_s.endWriteStruct()
    modality_s.startWriteStruct("standard_deviations", cv2.FileNode_SEQ)
    modality_s.write("", 15.0)
    modality_s.write("", 5.0)
    modality_s.write("", 3.5)
    modality_s.write("", 1.5)
    modality_s.endWriteStruct()
    modality_s.write("n_histogram_bins", 32)
    modality_s.write("learning_rate_f", 0.2)
    modality_s.write("learning_rate_b", 0.2)
    modality_s.write("unconsidered_line_length", 1.0)
    modality_s.write("max_considered_line_length", 18.0)
    modality_s.write("modeled_depth_offset_radius", 0.0)
    modality_s.write("modeled_occlusion_radius", 0.0)
    modality_s.write("modeled_occlusion_threshold", 0.03)
    # vis
    modality_s.write("visualize_pose_result", 0)
    modality_s.write("visualize_gradient_optimization", 0)
    modality_s.write("visualize_hessian_optimization", 0)
    modality_s.write("visualize_lines_correspondence", 0)
    modality_s.write("visualize_points_correspondence", 0)
    modality_s.write("visualize_points_depth_image_correspondence", 0)
    modality_s.write("visualize_points_depth_rendering_correspondence", 0)
    modality_s.release()

    config_yaml_path = os.path.join(icg_dir, "depth_modality.yaml")
    modality_s = cv2.FileStorage(config_yaml_path, cv2.FileStorage_WRITE)
    modality_s.write("visualize_pose_result", 0)
    modality_s.write("visualize_gradient_optimization", 0)
    modality_s.write("visualize_hessian_optimization", 0)
    modality_s.write("visualize_correspondences_correspondence", 0)
    modality_s.write("visualize_points_correspondence", 0)
    modality_s.write("visualize_points_depth_rendering_correspondence", 0)
    modality_s.write("visualization_max_depth", 2.0)  # the max depth for visualization
    modality_s.startWriteStruct("considered_distances", cv2.FileNode_SEQ)
    modality_s.write("", 0.05 * OBJECT_SCALE)
    modality_s.write("", 0.02 * OBJECT_SCALE)
    modality_s.write("", 0.01 * OBJECT_SCALE)
    modality_s.endWriteStruct()
    modality_s.release()

    config_yaml_path = os.path.join(icg_dir, "feature_modality.yaml")
    modality_s = cv2.FileStorage(config_yaml_path, cv2.FileStorage_WRITE)
    modality_s.write("visualize_pose_result", 0)
    modality_s.write("visualize_gradient_optimization", 0)
    modality_s.write("visualize_hessian_optimization", 0)
    modality_s.write("visualize_correspondences_correspondence", 0)
    modality_s.write("visualize_points_correspondence", 0)
    modality_s.write("visualize_points_depth_rendering_correspondence", 0)
    modality_s.write("visualize_points_result", 0)
    modality_s.write("visualization_max_depth", 2.0)  # the max depth for visualization
    modality_s.startWriteStruct("considered_distances", cv2.FileNode_SEQ)
    modality_s.write("", 0.05 * OBJECT_SCALE)
    modality_s.write("", 0.02 * OBJECT_SCALE)
    modality_s.write("", 0.01 * OBJECT_SCALE)
    modality_s.endWriteStruct()
    modality_s.write("port", feature_port)
    modality_s.write("roi_margin", 5)
    # match config
    modality_s.write("sfm_path", "/home/robot-learning/Projects/SuperPose/data/sfm_model/bakingsoda/outputs_superpoint_superglue")
    modality_s.release()

    # save the viewer
    config_yaml_path = os.path.join(icg_dir, "feature_viewer.yaml")
    viewer_s = cv2.FileStorage(config_yaml_path, cv2.FileStorage_WRITE)
    viewer_s.write("opacity", 0.9)
    viewer_s.write("display_images", 1)
    viewer_s.write("save_images", 0)
    viewer_s.write("save_directory", os.path.join(icg_dir, "debug"))
    viewer_s.write("save_image_type", "png")
    viewer_s.release()
    if not os.path.exists(os.path.join(icg_dir, "debug")):
        os.mkdir(os.path.join(icg_dir, "debug"))

    # save the object
    config_yaml_path = os.path.join(icg_dir, "object.yaml")
    object_s = cv2.FileStorage(config_yaml_path, cv2.FileStorage_WRITE)
    object_s.write("geometry_path", f"{tracker_name}.obj")
    object_s.write("geometry_unit_in_meter", 0.001)  # rbot model is in mm
    object_s.write("geometry_counterclockwise", 1)
    object_s.write("geometry_enable_culling", 0)
    object_s.write("geometry_enable_texture", 1)  # enable color
    object_s.write("geometry2body_pose", np.eye(4))  # the pose of the geometry in the body frame
    object_s.release()

    # save the opt file
    config_yaml_path = os.path.join(icg_dir, "optimizer.yaml")
    opt_s = cv2.FileStorage(config_yaml_path, cv2.FileStorage_WRITE)
    # rbot-related param
    opt_s.write("tikhonov_parameter_rotation", 1000.0)
    opt_s.write("tikhonov_parameter_translation", 100000.0)
    opt_s.release()

    # save the refiner file
    config_yaml_path = os.path.join(icg_dir, "refiner.yaml")
    refiner_s = cv2.FileStorage(config_yaml_path, cv2.FileStorage_WRITE)
    # rbot-related param
    refiner_s.write("n_update_iterations", 0)
    refiner_s.write("n_corr_iterations", 1)
    refiner_s.release()

    # save the tracker file
    config_yaml_path = os.path.join(icg_dir, "tracker.yaml")
    tracker_s = cv2.FileStorage(config_yaml_path, cv2.FileStorage_WRITE)
    # rbot-related param
    tracker_s.write("n_update_iterations", 2)
    tracker_s.write("n_corr_iterations", 7)
    tracker_s.release()


def check_sparse_model():
    # check visualization of the sparse model
    # gedoestic camaera pose and model
    pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker_name", type=str, default="", help="name")
    parser.add_argument("--model_dir", type=str, default="", help="where the cad model exists")
    parser.add_argument("--eval_dir", type=str, default="", help="where the eval data exists")
    parser.add_argument("--icg_dir", type=str, default="", help="exported icg data folder")
    parser.add_argument("--enable_depth", action="store_true", help="enable depth modality")
    parser.add_argument("--enable_feature", action="store_true", help="enable feature modality")
    parser.add_argument("--use_network_detector", action="store_true", help="use network detector")
    parser.add_argument("--detector_port", type=int, default=8080, help="detector port")
    parser.add_argument("--feature_port", type=int, default=9090, help="feature port")
    parser.add_argument("--enable_refine", action="store_true", help="enable refine")
    parser.add_argument("--redo_obj", action="store_true", help="redo the obj file")
    args = parser.parse_args()
    # generate the icg tracker config
    generate_icg_tracker(
        args.tracker_name,
        args.model_dir,
        args.eval_dir,
        args.icg_dir,
        args.use_network_detector,
        args.detector_port,
        args.feature_port,
        args.enable_depth,
        args.enable_feature,
        args.enable_refine,
    )
