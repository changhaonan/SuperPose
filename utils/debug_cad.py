import open3d as o3d
import glob
import os


def show_cad_model(model_dir):
    cad_file = glob.glob(os.path.join(model_dir, "*.obj"))[0]
    # read the mesh using open3d with texture
    cad_model = o3d.io.read_triangle_mesh(cad_file, True)
    # draw origin
    origin_frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=100.0)
    o3d.visualization.draw_geometries([cad_model, origin_frame])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="", help="path to model")
    args = parser.parse_args()

    # visualize cad model
    show_cad_model(args.model_dir)
