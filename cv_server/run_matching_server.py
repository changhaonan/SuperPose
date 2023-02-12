import cv2
import numpy as np
import zmq
import os
import tqdm


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
        # extract keypoints/descriptors for a single image
        xys_query, desc_query, scores_query = self.extractor.extract(image_query, 0)
        xys_train, desc_train, scores_train = self.extractor.extract(image_train, 1)

        if self.feature_type == "orb":
            # change to CV_8U
            desc_query = desc_query.astype(np.uint8)
            desc_train = desc_train.astype(np.uint8)
            # match
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(desc_query, desc_train)
        elif self.feature_type == "r2d2" or self.feature_type == "sift" or self.feature_type == "super_point":
            # change to CV_32F
            desc_query = desc_query.astype(np.float32)
            desc_train = desc_train.astype(np.float32)
            # match
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = bf.match(desc_query, desc_train)
        else:
            raise NotImplementedError
        matches = sorted(matches, key=lambda x: x.distance)

        # filter
        # matches = [m for m in matches if m.distance < threshold * 100]

        # visualize
        if len(matches) == 0:
            return []

        img_vis = cv2.drawMatches(
            image_query,
            [cv2.KeyPoint(x, y, 1) for x, y in xys_query],
            image_train,
            [cv2.KeyPoint(x, y, 1) for x, y in xys_train],
            matches[:],
            None,
            flags=2,
        )
        cv2.imshow("img", img_vis)
        cv2.waitKey(0)

        return matches


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
