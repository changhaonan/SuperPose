import cv2
import numpy as np
import zmq
import os
import tqdm
import matplotlib.pyplot as plt
import kornia as K
import kornia.feature as KF
import torch
from kornia_moons.feature import *


def to_torch_image(frame):
    img = K.image_to_tensor(frame).float() / 255.0
    img = K.color.bgr_to_rgb(img)
    # pad one dimension to make it a batch
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    img = img.cuda()
    return img


def get_matching_keypoints(lafs1, lafs2, idxs):
    mkpts1 = KF.get_laf_center(lafs1).squeeze()[idxs[:, 0]].detach().cpu().numpy()
    mkpts2 = KF.get_laf_center(lafs2).squeeze()[idxs[:, 1]].detach().cpu().numpy()
    return mkpts1, mkpts2


class NetworkExtractor:
    def __init__(self, port):
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://localhost:{self.port}")
        print("connected to server")

    def extract(self, img, frame_idx):
        # send the image as a multipart message
        msg0 = (
            np.array([img.shape[0], img.shape[1]])
            .reshape(-1)
            .astype(np.int32)
            .tobytes()
        )
        msg1 = img.astype(np.uint8).reshape(-1).tobytes()
        self.socket.send_multipart([msg0, msg1], 0)
        # receive the keypoints
        msgs = self.socket.recv_multipart(0)
        assert len(msgs) == 3, "#msgs={}".format(len(msgs))
        num_feat = np.frombuffer(msgs[0], dtype=np.int32)[0]
        feat_dim = np.frombuffer(msgs[0], dtype=np.int32)[1]
        xys = np.frombuffer(msgs[1], dtype=np.float32)
        if xys.shape[0] == num_feat * 2:
            xys = xys.reshape(num_feat, 2)
        elif xys.shape[0] == num_feat * 3:
            xys = xys.reshape(num_feat, 3)[:, :2]
        else:
            raise ValueError("xys.shape[0]={}".format(xys.shape[0]))
        desc = np.frombuffer(msgs[2], dtype=np.float32).reshape(num_feat, feat_dim)
        scores = np.ones(num_feat)
        return xys, desc, scores


class FeatureMatcher:
    def __init__(self, extractor, feature_type="orb") -> None:
        self.extractor = extractor
        self.feature_type = feature_type
        # set score scale
        if self.feature_type == "orb":
            self.score_scale = 1
            self.distance_scale = 1
        elif self.feature_type == "r2d2":
            self.score_scale = 1
            self.distance_scale = 1
        elif self.feature_type == "sift":
            self.score_scale = 1
            self.distance_scale = 1
        elif self.feature_type == "super_point":
            self.score_scale = 1
            self.distance_scale = 1

    def match(self, image_query, image_train, threshold=0.7):
        # process image
        image_query = to_torch_image(image_query)
        image_train = to_torch_image(image_train)

        # matching with loftr
        matcher = KF.LoFTR(pretrained='outdoor').eval().cuda()
        input_dict = {"image0": K.color.rgb_to_grayscale(image_query), # LofTR works on grayscale images only 
                    "image1": K.color.rgb_to_grayscale(image_train)}

        with torch.inference_mode():
            correspondences = matcher(input_dict)

        mkpts0 = correspondences['keypoints0'].cpu().numpy()
        mkpts1 = correspondences['keypoints1'].cpu().numpy()
        Fm, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
        inliers = inliers > 0

        if inliers.shape[0] == 0:  # Early return
            print("No inliers found")
            return 
        axis = draw_LAF_matches(
            KF.laf_from_center_scale_ori(torch.from_numpy(mkpts0).view(1,-1, 2),
                                        torch.ones(mkpts0.shape[0]).view(1,-1, 1, 1),
                                        torch.ones(mkpts0.shape[0]).view(1,-1, 1)),

            KF.laf_from_center_scale_ori(torch.from_numpy(mkpts1).view(1,-1, 2),
                                        torch.ones(mkpts1.shape[0]).view(1,-1, 1, 1),
                                        torch.ones(mkpts1.shape[0]).view(1,-1, 1)),
            torch.arange(mkpts0.shape[0]).view(-1,1).repeat(1,2),
            K.tensor_to_image(image_query),
            K.tensor_to_image(image_train),
            inliers,
            draw_dict={'inlier_color': (0.2, 1, 0.2),
                    'tentative_color': None, 
                    'feature_color': (0.2, 0.5, 1), 'vertical': False})
        plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="orb", help="feature type")
    parser.add_argument("--save_dir", type=str, default=None, help="save directory")
    parser.add_argument(
        "--query_dir", type=str, default=None, help="query image directory"
    )
    parser.add_argument(
        "--train_dir", type=str, default=None, help="train image directory"
    )
    parser.add_argument("--port", type=int, default=9090, help="port number")
    parser.add_argument("--top-k", type=int, default=500, help="number of keypoints")
    args = parser.parse_args()

    # prepare path
    if args.save_dir is not None:
        if os.path.exists(args.save_dir) is False:
            os.makedirs(args.save_dir)

    extractor = NetworkExtractor(args.port)
    matcher = FeatureMatcher(extractor, args.type)

    # Go through all images in the query and train directories
    for query_image_name in tqdm.tqdm(os.listdir(args.query_dir)):
        query_image = cv2.imread(os.path.join(args.query_dir, query_image_name))
        for train_image_name in tqdm.tqdm(os.listdir(args.train_dir)):
            train_image = cv2.imread(os.path.join(args.train_dir, train_image_name))
            matches = matcher.match(query_image, train_image)
