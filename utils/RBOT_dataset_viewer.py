import open3d as o3d
import os

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="/home/robot-learning/Projects/SuperPose/data/RBOT_dataset")
    parser.add_argument("--scene_name", type=str, default="bakingsoda")
    args = parser.parse_args()

    # read the obj, texture, and obj.mtl files
    obj_path = os.path.join(args.dataset_dir, args.scene_name, f"{args.scene_name}.obj")
    texture_path = os.path.join(args.dataset_dir, args.scene_name, f"{args.scene_name}_tex.png")
    mtl_path = os.path.join(args.dataset_dir, args.scene_name, f"{args.scene_name}.obj.mtl")

    # read the obj file
    mesh = o3d.io.read_triangle_mesh(obj_path, True)
    
    # save the obj
    output_path = os.path.join(args.dataset_dir, args.scene_name, f"{args.scene_name}_output.obj")
    o3d.io.write_triangle_mesh(output_path, mesh)

    # visualize
    # o3d.visualization.draw_geometries([mesh])
