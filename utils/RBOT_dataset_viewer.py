import open3d as o3d
import os
import cv2

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="/home/robot-learning/Projects/SuperPose/ICG/examples/generator_example_v2")
    parser.add_argument("--scene_name", type=str, default="bakingsoda")
    args = parser.parse_args()

    # read the obj, texture, and obj.mtl files
    # obj_path = os.path.join(args.dataset_dir, args.scene_name, f"{args.scene_name}.obj")
    # texture_path = os.path.join(args.dataset_dir, args.scene_name, f"{args.scene_name}_tex.png")
    # mtl_path = os.path.join(args.dataset_dir, args.scene_name, f"{args.scene_name}.obj.mtl")

    obj_path = os.path.join(args.dataset_dir, f"{args.scene_name}.obj")
    texture_path = os.path.join(args.dataset_dir, f"{args.scene_name}_tex.png")
    mtl_path = os.path.join(args.dataset_dir, f"{args.scene_name}.obj.mtl")

    # read the obj file
    mesh = o3d.io.read_triangle_mesh(obj_path, True)
    
    # # rotate the texture image by 90 degrees
    # texture = cv2.imread(texture_path)
    # texture = cv2.rotate(texture, cv2.ROTATE_90_CLOCKWISE)
    # cv2.imwrite(texture_path, texture)

    # flip the texture image vertically
    # texture = cv2.imread(texture_path)
    # texture = cv2.flip(texture, 0)
    # cv2.imwrite(texture_path, texture)

    # save the obj
    # output_path = os.path.join(args.dataset_dir, args.scene_name, f"{args.scene_name}_output.obj")
    # o3d.io.write_triangle_mesh(output_path, mesh)

    # create origin
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0, origin=[0, 0, 0]) 

    # # visualize
    o3d.visualization.draw_geometries([mesh, origin])
