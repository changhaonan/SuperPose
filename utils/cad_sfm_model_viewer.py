import open3d as o3d
import numpy as np
import os
import struct
import collections
import json

Point3D = collections.namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])


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
            binary_point_line_properties = read_next_bytes(fid, num_bytes=43, format_char_sequence="QdddBBBd")
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(fid, num_bytes=8 * track_length, format_char_sequence="ii" * track_length)
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(id=point3D_id, xyz=xyz, rgb=rgb, error=error, image_ids=image_ids, point2D_idxs=point2D_idxs)
    return points3D


# transfer colmap Points3D to open3d PointCloud:
def points_colmap_to_o3d(points3D):
    points3D_list = []
    points3D_color = []
    for k, v in points3D.items():
        points3D_list.append(v.xyz)
        points3D_color.append(v.rgb)
    points_o3d = o3d.geometry.PointCloud()
    points_o3d.points = o3d.utility.Vector3dVector(np.stack(points3D_list, axis=0))
    points_o3d.colors = o3d.utility.Vector3dVector(np.stack(points3D_color, axis=0) / 255.0)
    return points_o3d


def generate_bbox_transformed(box3d_path, cam_poses):
    box3d = np.loadtxt(box3d_path)
    # create bbox linset from bbox points
    bbox = o3d.geometry.LineSet()
    bbox.points = o3d.utility.Vector3dVector(box3d)
    bbox.lines = o3d.utility.Vector2iVector([[0, 1], [1, 2], [2, 3], [3, 0]])
    bbox.colors = o3d.utility.Vector3dVector([[1, 0, 0] for i in range(len(bbox.lines))])
    # transform bbox to camera poses
    bbox_list = []
    for i in range(len(cam_poses)):
        bbox_i = bbox.transform(cam_poses[i])
        bbox_list.append(bbox_i)
    return bbox_list


def vis_pcd(pcd, cam_poses=None, num_inliers=None, coord_frame_size=0.2):
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
        for i in range(len(cam_poses)):
            cam_pose = cam_poses[i]
            if num_inliers is not None:
                num_inlier = num_inliers[i]
                if num_inlier > 10:
                    cam_frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=coord_frame_size)
                    cam_frame.transform(cam_pose)
                    vis_list.append(cam_frame)
            else:
                cam_frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=coord_frame_size)
                cam_frame.transform(cam_pose)
                vis_list.append(cam_frame)
        o3d.visualization.draw_geometries(vis_list)
    else:
        o3d.visualization.draw_geometries(vis_list)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", type=str, default="", help="path to model")
    parser.add_argument("--sparse_model_path", type=str, default="", help="path to sparse model")
    args = parser.parse_args()

    # read the mesh using open3d
    mesh = o3d.io.read_triangle_mesh(args.model_file, False)
    mesh.scale(1.0 / 1000.0, center=np.zeros(3, dtype=np.float64))
    # create the origin coordinate frame
    origin_frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=1.0)

    # read the sparse model
    sparse_model_path = os.path.join(args.sparse_model_path, "outputs_superpoint_superglue/sfm_ws/model/points3D.bin")
    points_3d = read_points3d_binary(sparse_model_path)
    sprase_model = points_colmap_to_o3d(points_3d)

    # read intrinsics
    intrinsics_path = os.path.join(args.sparse_model_path, "intrin_ba/0.txt")
    intrinsics = np.loadtxt(intrinsics_path)

    # create camera intrinsics
    img_height = int(intrinsics[1, 1] * 2)
    img_width = int(intrinsics[0, 0] * 2)
    cam_intrinsic = o3d.camera.PinholeCameraIntrinsic(img_width, img_height, intrinsics)

    # read camera poses
    camera_poses_path = os.path.join(args.sparse_model_path, "poses_ba")
    cam_poses = []
    for i in range(10):
        cam_pose_path = os.path.join(camera_poses_path, f"{i}.txt")
        cam_pose = np.loadtxt(cam_pose_path)
        cam_poses.append(np.linalg.inv(cam_pose))

    # read depth image
    depth_path = os.path.join(args.sparse_model_path, "depth")
    depth_pcd_list = []
    for check_idx in range(10):
        depth_image = o3d.io.read_image(os.path.join(depth_path, f"{check_idx}.png"))
        # create point cloud from depth image
        depth_pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_image, cam_intrinsic, np.linalg.inv(cam_poses[check_idx]))
        depth_pcd_list.append(depth_pcd)

    vis_pcd([mesh, sprase_model, *depth_pcd_list], cam_poses, coord_frame_size=0.1)
