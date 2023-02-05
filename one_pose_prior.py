"""
Generate 6D pose priors using OnePose
"""
import cv2
import os
import cv2
import numpy as np
import hydra
from one_pose_inference import OnePoseInference

@hydra.main(config_path='cfg/', config_name='config.yaml')
def generate_prior(cfg):
    data_dirs = cfg.input.data_dirs
    eval_dirs = cfg.input.eval_dirs
    sfm_model_dirs = cfg.input.sfm_model_dirs
    if isinstance(data_dirs, str) and isinstance(sfm_model_dirs, str):
        data_dirs = [data_dirs]
        sfm_model_dirs = [sfm_model_dirs]
    # get the first one
    sfm_data_dir = os.path.join(data_dirs[0].split(" ")[0], data_dirs[0].split(" ")[1])
    sfm_model_dir = sfm_model_dirs[0]
    # init one pose inference
    one_pose_inference = OnePoseInference(cfg, sfm_data_dir, sfm_model_dir) 
    
    # hard-code
    video_mode = cfg.video_mode
    for eval_dir in eval_dirs:
        # generate one pose prior
        pose_pred_homo_list = []
        num_inliers_list = []
        frame = 0
        if video_mode == "video":
            cap = cv2.VideoCapture(f"{eval_dir}/color.avi")
            while True:
                ret, frame = cap.read()
                if ret:
                    pose_pred, pose_pred_homo, num_inliers = one_pose_inference.inference(cfg, frame)
                    pose_pred_homo_list.append(np.linalg.inv(pose_pred_homo))
                    num_inliers_list.append(num_inliers)
                    frame += 1
                else:
                    break
        elif video_mode == "image":
            cap = cv2.VideoCapture(f"{eval_dir}/color/%d.png")
            while True:
                ret, frame = cap.read()
                if ret:
                    pose_pred, pose_pred_homo, num_inliers = one_pose_inference.inference(cfg, frame, True)
                    pose_pred_homo_list.append(np.linalg.inv(pose_pred_homo))
                    num_inliers_list.append(num_inliers)
                    frame += 1
                else:
                    break
        else:
            print(f"Video mode {video_mode} not supported.")
        # save the prior into kf_results.npz
        pose_pred_homo_list = np.stack(pose_pred_homo_list, axis=0)
        num_inliers_list = np.stack(num_inliers_list, axis=0)
        np.savez(f"{eval_dir}/kf_results.npz", cam_poses=pose_pred_homo_list, num_inliers=num_inliers_list)


if __name__ == "__main__":
    generate_prior()