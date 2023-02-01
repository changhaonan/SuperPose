import json
import os
import glob
import hydra
import numpy as np

import os.path as osp
from loguru import logger
from pathlib import Path
from omegaconf import DictConfig


def merge_(anno_2d_file, avg_anno_3d_file, collect_anno_3d_file,
           idxs_file, img_id, ann_id, images, annotations):
    """ To prepare training and test objects, we merge annotations about difference objs"""
    with open(anno_2d_file, 'r') as f:
        annos_2d = json.load(f)
    
    for anno_2d in annos_2d:
        img_id += 1
        info = {
            'id': img_id,
            'img_file': anno_2d['img_file'],
        }
        images.append(info)

        ann_id += 1
        anno = {
            'image_id': img_id,
            'id': ann_id,
            'pose_file': anno_2d['pose_file'],
            'anno2d_file': anno_2d['anno_file'],
            'avg_anno3d_file': avg_anno_3d_file,
            'collect_anno3d_file': collect_anno_3d_file,
            'idxs_file': idxs_file
        }
        annotations.append(anno)
    return img_id, ann_id


def merge_anno(cfg):
    """ Merge different objects' anno file into one anno file """
    anno_dirs = []

    if cfg.split == 'train':
        names = cfg.train.names
    elif cfg.split == 'val':
        names = cfg.val.names
    
    for name in names:
        anno_dir = osp.join(cfg.datamodule.data_dir, name, f'outputs_{cfg.network.detection}_{cfg.network.matching}', 'anno')
        anno_dirs.append(anno_dir) 
    
    img_id = 0
    ann_id = 0
    images = []
    annotations = []
    for anno_dir in anno_dirs:
        logger.info(f'Merging anno dir: {anno_dir}')
        anno_2d_file = osp.join(anno_dir, 'anno_2d.json')
        avg_anno_3d_file = osp.join(anno_dir, 'anno_3d_average.npz')
        collect_anno_3d_file = osp.join(anno_dir, 'anno_3d_collect.npz')
        idxs_file = osp.join(anno_dir, 'idxs.npy')

        if not osp.isfile(anno_2d_file) or not osp.isfile(avg_anno_3d_file) or not osp.isfile(collect_anno_3d_file):
            logger.info(f'No annotation in: {anno_dir}')
            continue
        
        img_id, ann_id = merge_(anno_2d_file, avg_anno_3d_file, collect_anno_3d_file,
                                idxs_file, img_id, ann_id, images, annotations)
    
    logger.info(f'Total num: {len(images)}')
    instance = {'images': images, 'annotations': annotations}

    out_dir = osp.dirname(cfg.datamodule.out_path)
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    with open(cfg.datamodule.out_path, 'w') as f:
        json.dump(instance, f)


def sfm(cfg):
    """ Reconstruct and postprocess sparse object point cloud, and store point cloud features"""
    from src.utils import path_utils

    data_dirs = cfg.dataset.data_dir
    down_ratio = cfg.sfm.down_ratio
    data_dirs = [data_dirs] if isinstance(data_dirs, str) else data_dirs
    
    for data_dir in data_dirs:
        logger.info(f"Processing {data_dir}.")
        root_dir, sub_dirs = data_dir.split(' ')[0], data_dir.split(' ')[1:]

        # Parse image directory and downsample images:
        img_lists = []
        seg_lists = []
        for sub_dir in sub_dirs:
            seq_dir = osp.join(root_dir, sub_dir)
            img_lists += glob.glob(str(Path(seq_dir)) + '/color/*.png', recursive=True)
            seg_lists += glob.glob(str(Path(seq_dir)) + '/seg/*.png', recursive=True)

        # Filter out images based on the existence of pose file:
        pose_lists = [path_utils.get_gt_pose_path_by_color(color_path) for color_path in img_lists]
        img_lists = [img for img, pose in zip(img_lists, pose_lists) if osp.isfile(pose)]
        seg_lists = [seg for seg, pose in zip(seg_lists, pose_lists) if osp.isfile(pose)]

        # Sorted img_lists:
        img_lists = sorted(img_lists, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        seg_lists = sorted(seg_lists, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        down_img_lists = []
        down_seg_lists = []
        for seg_file, img_file in zip(seg_lists, img_lists):
            index = int(img_file.split('/')[-1].split('.')[0])
            if index % down_ratio == 0:
                down_img_lists.append(img_file)  
                down_seg_lists.append(seg_file)

        img_lists = down_img_lists
        seg_lists = down_seg_lists
        if len(img_lists) == 0 or len(seg_lists) == 0:
            logger.info(f"No png image in {root_dir}")
            continue
        
        obj_name = root_dir.split('/')[-1]
        outputs_dir_root = cfg.dataset.outputs_dir.format(obj_name)

        # Begin SfM and postprocess:
        sfm_core(cfg, img_lists, seg_lists, outputs_dir_root)
        postprocess(cfg, img_lists, root_dir, outputs_dir_root) 


def sfm_core(cfg, img_lists, seg_lists, outputs_dir_root):
    """ Sparse reconstruction: extract features, match features, triangulation"""
    from src.sfm import extract_features_ext, match_features, \
                         generate_empty, triangulation, pairs_from_poses

    # Construct output directory structure:
    outputs_dir = osp.join(outputs_dir_root, 'outputs' + '_' + cfg.network.detection + '_' + cfg.network.matching)
    feature_out = osp.join(outputs_dir, f'feats-{cfg.network.detection}.h5')
    covis_pairs_out = osp.join(outputs_dir, f'pairs-covis{cfg.sfm.covis_num}.txt')
    matches_out = osp.join(outputs_dir, f'matches-{cfg.network.matching}.h5')
    empty_dir = osp.join(outputs_dir, 'sfm_empty')
    deep_sfm_dir = osp.join(outputs_dir, 'sfm_ws')
    
    if cfg.redo:
        os.system(f'rm -rf {outputs_dir}') 
        Path(outputs_dir).mkdir(exist_ok=True, parents=True)

        # Extract image features, construct image pairs and then match:
        extract_features_ext.main(img_lists, seg_lists, feature_out, cfg)
        pairs_from_poses.covis_from_pose(img_lists, covis_pairs_out, cfg.sfm.covis_num, max_rotation=cfg.sfm.rotation_thresh)
        match_features.main(cfg, feature_out, covis_pairs_out, matches_out, vis_match=False)

        # Reconstruct 3D point cloud with known image poses:
        generate_empty.generate_model(img_lists, empty_dir)
        triangulation.main(deep_sfm_dir, empty_dir, outputs_dir, covis_pairs_out, feature_out, matches_out, image_dir=None)

    
def postprocess(cfg, img_lists, root_dir, outputs_dir_root):
    """ Filter points and average feature"""
    from src.sfm.postprocess import filter_points, feature_process, filter_tkl
    from src.utils.colmap import read_write_model 
    from src.utils.geo_utils import points_colmap_to_o3d, bbox_o3d_to_onepose
    import open3d as o3d

    bbox_path = osp.join(root_dir, "box3d_corners.txt")
    # Construct output directory structure:
    outputs_dir = osp.join(outputs_dir_root, 'outputs' + '_' + cfg.network.detection + '_' + cfg.network.matching)
    feature_out = osp.join(outputs_dir, f'feats-{cfg.network.detection}.h5')
    deep_sfm_dir = osp.join(outputs_dir, 'sfm_ws')
    model_path = osp.join(deep_sfm_dir, 'model')

    # Generate bounding box using extracted features:
    model_path = osp.join(deep_sfm_dir, 'model')
    # Read points3D.bin
    points_model_path = osp.join(model_path, 'points3D.bin')
    points3D_colmap = read_write_model.read_points3d_binary(points_model_path)
    points3D_o3d = points_colmap_to_o3d(points3D_colmap)
    cl, ind = points3D_o3d.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    # Run DBSCAN to filter noisy points
    
    bbox_o3d = o3d.geometry.OrientedBoundingBox.create_from_points(cl.points)
    bbox_o3d.color = (1, 0, 0)
    bbox_corners = bbox_o3d.get_box_points()
    bbox_corners = bbox_o3d_to_onepose(np.asarray(bbox_corners))  # Convert to onepose format
    np.savetxt(bbox_path, bbox_corners)

    # Select feature track length to limit the number of 3D points below the 'max_num_kp3d' threshold:
    track_length, points_count_list = filter_tkl.get_tkl(model_path, thres=cfg.dataset.max_num_kp3d, show=False) 
    filter_tkl.vis_tkl_filtered_pcds(model_path, points_count_list, track_length, outputs_dir) # For visualization only

    # Leverage the selected feature track length threshold and 3D BBox to filter 3D points:
    xyzs, points_idxs = filter_points.filter_3d(model_path, track_length, bbox_path)
    # Merge 3d points by distance between points
    merge_xyzs, merge_idxs = filter_points.merge(xyzs, points_idxs, dist_threshold=1e-3) 

    # Save features of the filtered point cloud:
    feature_process.get_kpt_ann(cfg, img_lists, feature_out, outputs_dir, merge_idxs, merge_xyzs)
    

@hydra.main(config_path='configs/', config_name='config.yaml')
def main(cfg: DictConfig):
    globals()[cfg.type](cfg)


if __name__ == "__main__":
    main()