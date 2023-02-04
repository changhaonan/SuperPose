import open3d as o3d 
import numpy as np
import os
import struct
import collections
import json

Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_points3d_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id, xyz=xyz, rgb=rgb,
                error=error, image_ids=image_ids,
                point2D_idxs=point2D_idxs)
    return points3D


# transfer colmap Points3D to open3d PointCloud:
def points_colmap_to_o3d(points3D):
    points3D_list = []
    for k, v in points3D.items():
        points3D_list.append(v.xyz)
    points_o3d = o3d.geometry.PointCloud()
    points_o3d.points = o3d.utility.Vector3dVector(np.stack(points3D_list, axis=0))
    return points_o3d


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
    parser.add_argument("--sparse_model_path", type=str, default="", help="path to sparse model")
    args = parser.parse_args()

    # read the point cloud using open3d
    pcd = o3d.io.read_point_cloud(os.path.join(args.model_path, "reconstruct.pcd"))
    # read the mesh using open3d
    mesh = o3d.io.read_triangle_mesh(os.path.join(args.model_path, "reconstruct.obj"))
    # create the origin coordinate frame
    origin_frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.2)
    # read the camera pose
    kf_results = np.load(os.path.join(args.model_path, "kf_results.npz"))
    # load context 
    cam_poses = kf_results["cam_poses"]
    context_json_file = os.path.join(args.model_path, "context.json")
    with open(context_json_file, "r") as f:
        context_data = json.load(f)
    intrinsic = np.array([
        context_data["cam-00"]["intrinsic"][0], 
        context_data["cam-00"]["intrinsic"][1], 
        context_data["cam-00"]["intrinsic"][2], 
        context_data["cam-00"]["intrinsic"][3]])
    intrinsic_matrix = np.array([
        [intrinsic[0], 0, intrinsic[2]],
        [0, intrinsic[1], intrinsic[3]],
        [0, 0, 1]])
    # create camera intrinsics
    cam_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        context_data["cam-00"]["image_cols"], 
        context_data["cam-00"]["image_rows"], intrinsic_matrix)

    # read depth image
    depth_path = os.path.join(args.model_path, "../depth")
    depth_image = o3d.io.read_image(os.path.join(depth_path, "494.png"))
    # create point cloud from depth image
    depth_pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_image, cam_intrinsic, np.linalg.inv(cam_poses[-1]))

    if not args.sparse_model_path:
        # visualize the point cloud
        vis_pcd([pcd], kf_results["cam_poses"])
    else:
        # read the sparse model
        sparse_model_path = os.path.join(args.sparse_model_path, "points3D.bin")
        points_3d = read_points3d_binary(sparse_model_path)
        sprase_model = points_colmap_to_o3d(points_3d)
        # visualize the point cloud
        vis_pcd([depth_pcd, mesh, origin_frame], kf_results["cam_poses"])