from __future__ import print_function
import numpy as np
import zmq
import cv2
from tqdm import tqdm
import numpy as np
import os


class ORBExtractor:
    def __init__(self, top_k, save_dir):
        self.top_k = top_k
        self.save_dir = save_dir

    def extract(self, img, frame_idx):
        # extract keypoints/descriptors for a single image
        orb = cv2.ORB_create()
        kpts, desc = orb.detectAndCompute(img, None)
        xys = np.array([k.pt for k in kpts])
        scores = np.array([k.response for k in kpts])
        idxs = scores.argsort()[-self.top_k or None :]
        # visualize
        # img = cv2.drawKeypoints(img, kpts, None, color=(0, 255, 0), flags=0)
        # cv2.imshow('img', img)
        # cv2.waitKey(1)
        # if len(idxs) == 0:
        #     return np.zeros((0, 2)), np.zeros((0, 32)), np.zeros((0, ))
        # save the image
        if self.save_dir is not None:
            if os.path.exists(self.save_dir) is False:
                os.makedirs(self.save_dir)
            cv2.imwrite(os.path.join(self.save_dir, f"{frame_idx}.png"), img)
        return xys[idxs], desc[idxs], scores[idxs]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="r2d2", help="output file tag")
    parser.add_argument("--save-dir", type=str, default=None, help="data directory")
    parser.add_argument("--top-k", type=int, default=5000, help="number of keypoints")
    parser.add_argument("--scale-f", type=float, default=2**0.25)
    parser.add_argument("--min-size", type=int, default=256)
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument("--min-scale", type=float, default=0)
    parser.add_argument("--max-scale", type=float, default=1)

    parser.add_argument("--reliability-thr", type=float, default=0.7)
    parser.add_argument("--repeatability-thr", type=float, default=0.7)

    parser.add_argument(
        "--gpu", type=int, nargs="+", default=[0], help="use -1 for CPU"
    )
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args()

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    port = f"tcp://*:{args.port}"
    print("port", port)
    socket.bind(port)

    # init extractor
    extractor = ORBExtractor(args.top_k, args.save_dir)
    frame_idx = 0
    while True:
        print(f"ORB listending to {port}")
        msgs = socket.recv_multipart(0)
        assert len(msgs) == 2, "#msgs={}".format(len(msgs))
        wh = np.frombuffer(msgs[0], dtype=np.int32)
        W = wh[0]
        H = wh[1]
        print(f"W={W}, H={H}")
        msg = msgs[1]
        image = np.frombuffer(msg, dtype=np.uint8).reshape(H, W, -1).squeeze()

        # extract keypoints/descriptors
        xys, desc, scores = extractor.extract(image, frame_idx)

        num_feat = len(xys)
        feat_dim = desc.shape[1]
        print(f"num_feat={num_feat}, feat_dim={feat_dim}.")
        if num_feat == 0:
            msg = np.array([0, 0]).reshape(-1).astype(np.int32).tobytes()
            socket.send(msg, 0)
            continue
        msg = np.array([num_feat, feat_dim]).reshape(-1).astype(np.int32).tobytes()
        socket.send(msg, 2)
        msg = xys.astype(np.float32).reshape(-1).tobytes()
        socket.send(msg, 2)
        msg = desc.astype(np.float32).reshape(-1).tobytes()
        socket.send(msg, 0)

        # update frame index
        frame_idx += 1
