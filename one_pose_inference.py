"""
OnePose inference, but online
"""
import os
import cv2
import glob
import torch
import hydra
from tqdm import tqdm
import os.path as osp
import numpy as np

from PIL import Image
from loguru import logger
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from one_pose.utils import data_utils, path_utils, eval_utils, vis_utils

from pytorch_lightning import seed_everything
seed_everything(12345)


def get_default_paths(cfg, data_dir, sfm_model_dir):
    data_root = os.path.dirname(data_dir)
    anno_dir = osp.join(sfm_model_dir, f'outputs_{cfg.network.detection}_{cfg.network.matching}', 'anno')
    avg_anno_3d_path = osp.join(anno_dir, 'anno_3d_average.npz')
    clt_anno_3d_path = osp.join(anno_dir, 'anno_3d_collect.npz')
    idxs_path = osp.join(anno_dir, 'idxs.npy')

    object_detect_mode = cfg.object_detect_mode
    logger.info(f"Use {object_detect_mode} as object detector")


    intrin_full_path = osp.join(data_dir, 'intrinsics.txt')
    paths = {
        'data_dir': data_dir,
        'data_root': data_root,
        'sfm_model_dir': sfm_model_dir,
        'avg_anno_3d_path': avg_anno_3d_path,
        'clt_anno_3d_path': clt_anno_3d_path,
        'idxs_path': idxs_path,
        'intrin_full_path': intrin_full_path
    }
    return paths


def load_model(cfg):
    """ Load model """
    def load_matching_model(model_path):
        """ Load onepose model """
        from one_pose.models.GATsSPG_lightning_model import LitModelGATsSPG

        trained_model = LitModelGATsSPG.load_from_checkpoint(checkpoint_path=model_path)
        trained_model.cuda()
        trained_model.eval()
        trained_model.freeze()

        return trained_model

    def load_extractor_model(cfg, model_path):
        """ Load extractor model(SuperPoint) """
        from one_pose.models.extractors.SuperPoint.superpoint import SuperPoint
        from one_pose.sfm.extract_features import confs
        from one_pose.utils.model_io import load_network

        extractor_model = SuperPoint(confs[cfg.network.detection]['conf'])
        extractor_model.cuda()
        extractor_model.eval()
        load_network(extractor_model, model_path)

        return extractor_model

    matching_model = load_matching_model(cfg.model.onepose_model_path)
    extractor_model = load_extractor_model(cfg, cfg.model.extractor_model_path)
    return matching_model, extractor_model


def pack_data(avg_descriptors3d, clt_descriptors, keypoints3d, detection, image_size):
    """ Prepare data for OnePose inference """
    keypoints2d = torch.Tensor(detection['keypoints'])
    descriptors2d = torch.Tensor(detection['descriptors'])

    inp_data = {
        'keypoints2d': keypoints2d[None].cuda(), # [1, n1, 2]
        'keypoints3d': keypoints3d[None].cuda(), # [1, n2, 3]
        'descriptors2d_query': descriptors2d[None].cuda(), # [1, dim, n1]
        'descriptors3d_db': avg_descriptors3d[None].cuda(), # [1, dim, n2]
        'descriptors2d_db': clt_descriptors[None].cuda(), # [1, dim, n2*num_leaf]
        'image_size': image_size
    }
    return inp_data


def load_intrinsic(intrin_full_path):
    """ Load intrinsic matrix """
    with open(intrin_full_path, 'r') as f:
        intrin = f.readlines()
    K = np.eye(3)
    K[0, 0] = float(intrin[0].split(' ')[-1])
    K[1, 1] = float(intrin[1].split(' ')[-1])
    K[0, 2] = float(intrin[2].split(' ')[-1])
    K[1, 2] = float(intrin[3].split(' ')[-1])
    return K


class OnePoseInference:
    @torch.no_grad()
    def __init__(self, cfg, data_dir, sfm_model_dir) -> None:
        from one_pose.datasets.normalized_dataset import NormalizedDataset
        from one_pose.sfm.extract_features import confs
        from one_pose.evaluators.cmd_evaluator import Evaluator

        self.matching_model, self.extractor_model = load_model(cfg)
        self.paths = get_default_paths(cfg, data_dir, sfm_model_dir)

        num_leaf = cfg.num_leaf
        avg_data = np.load(self.paths['avg_anno_3d_path'])
        clt_data = np.load(self.paths['clt_anno_3d_path'])
        idxs = np.load(self.paths['idxs_path'])

        self.keypoints3d = torch.Tensor(clt_data['keypoints3d']).cuda()
        self.num_3d = self.keypoints3d.shape[0]
        # Load average 3D features:
        self.avg_descriptors3d, _ = data_utils.pad_features3d_random(
                                    avg_data['descriptors3d'],
                                    avg_data['scores3d'],
                                    self.num_3d
                                )
        # Load corresponding 2D features of each 3D point:
        self.clt_descriptors, _ = data_utils.build_features3d_leaves(
                                    clt_data['descriptors3d'],
                                    clt_data['scores3d'],
                                    idxs, self.num_3d, num_leaf
                                )

    @torch.no_grad()
    def inference(self, cfg, image, enable_vis=False):
        # Normalize image:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        inp = transforms.ToTensor()(image_gray).cuda()[None]
        intrin_path = self.paths["intrin_full_path"]
        K_crop = load_intrinsic(intrin_path)
        image_size = inp.shape[-2:]

        # Detect query image keypoints and extract descriptors:
        pred_detection = self.extractor_model(inp)
        pred_detection = {k: v[0].cpu().numpy() for k, v in pred_detection.items()}

        # 2D-3D matching by GATsSPG:
        inp_data = pack_data(self.avg_descriptors3d, self.clt_descriptors, self.keypoints3d, pred_detection, image_size)
        pred, _ = self.matching_model(inp_data)
        matches = pred['matches0'].detach().cpu().numpy()
        valid = matches > -1
        kpts2d = pred_detection['keypoints']
        kpts3d = inp_data['keypoints3d'][0].detach().cpu().numpy()
        confidence = pred['matching_scores0'].detach().cpu().numpy()
        mkpts2d, mkpts3d, mconf = kpts2d[valid], kpts3d[matches[valid]], confidence[valid]

        # Estimate object pose by 2D-3D correspondences:
        pose_pred, pose_pred_homo, inliers = eval_utils.ransac_PnP(K_crop, mkpts2d, mkpts3d, scale=1000)
        
        # Visualize the result:
        if enable_vis and len(inliers) > 0:
            poses = [pose_pred_homo]
            box3d_path = path_utils.get_3d_box_path(self.paths['data_root'])
            intrin_full_path = path_utils.get_intrin_full_path(self.paths['data_dir'])
            # visualize bbox
            image_vis = vis_utils.vis_reproj_image(image, poses, box3d_path, intrin_full_path, colors=['y'])
            # visualize keypoints
            image_vis = vis_utils.vis_keypoints(image_vis, kpts2d[valid], color='g')
            image_vis = vis_utils.vis_keypoints(image_vis, kpts2d[~valid], color='r')
            # resize frame
            vis_heght = 640
            vis_width = int(image_vis.shape[1] * vis_heght / image_vis.shape[0])
            image_vis = cv2.resize(image_vis, (vis_width, vis_heght))
            cv2.imshow('frame', image_vis)
            cv2.waitKey(15)
        return pose_pred, pose_pred_homo, len(inliers)

@hydra.main(config_path='cfg/', config_name='config.yaml')
def main(cfg):
    data_dirs = cfg.input.data_dirs
    sfm_model_dirs = cfg.input.sfm_model_dirs
    if isinstance(data_dirs, str) and isinstance(sfm_model_dirs, str):
        data_dirs = [data_dirs]
        sfm_model_dirs = [sfm_model_dirs]
    # get the first one
    sfm_data_dir = os.path.join(data_dirs[0].split(" ")[0], data_dirs[0].split(" ")[1])
    sfm_model_dir = sfm_model_dirs[0]
    test_data_dir = os.path.join(data_dirs[0].split(" ")[0], data_dirs[0].split(" ")[2])
    # init one pose inference
    one_pose_inference = OnePoseInference(cfg, sfm_data_dir, sfm_model_dir) 

    if os.path.exists(f"{test_data_dir}/video.MOV"):
        video_mode = "video"
    elif os.path.exists(f"{test_data_dir}/images/1.png"):
        video_mode = "images"
    else:
        video_mode = "web_camera"
    
    # hard-code
    video_mode = "realsense"
    if video_mode == "video":
        cap = cv2.VideoCapture(f"{test_data_dir}/color.avi")
        while True:
            ret, frame = cap.read()
            if ret:
                one_pose_inference.inference(cfg, frame, True)
    elif video_mode == "web_camera":
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if ret:
                one_pose_inference.inference(cfg, frame, True)
    elif video_mode == "realsense":
        from utils.camera_utils import RealSenseCamera
        camera = RealSenseCamera()
        while True:
            depth_image, color_image = camera.get_frame()
            if depth_image is None or color_image is None:
                continue
            if color_image is not None:
                one_pose_inference.inference(cfg, color_image, True)
    elif video_mode == "images":
        cap = cv2.VideoCapture(f"{test_data_dir}/images/%d.png")
        while True:
            ret, frame = cap.read()
            if ret:
                one_pose_inference.inference(cfg, frame, True)


if __name__ == "__main__":
    main()
