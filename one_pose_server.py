import numpy as np
import cv2
import os
import cv2
import numpy as np
import hydra
import zmq
from one_pose_inference import OnePoseInference

def pose_to_string(pose):
    pose_list = pose.flatten().tolist()
    pose_string = ""
    for i in range(len(pose_list)):
        pose_string += str(pose_list[i])
        if i != len(pose_list) - 1:
            pose_string += " "
    return pose_string


class PoseServer:
    def __init__(self, pose_estimator, host='127.0.0.1', port=8080):
        self.pose_estimator = pose_estimator
        # Set up zmq
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        self.port = f"tcp://*:{port}"
        print("port", port)
        self.socket.bind(self.port)

    def run_service(self):
        print(f"OnePose listending to {self.port}")
        msgs = self.socket.recv_multipart(0)
        assert len(msgs) == 2, "#msgs={}".format(len(msgs))
        wh = np.frombuffer(msgs[0], dtype=np.int32)
        W = wh[0]
        H = wh[1]
        print(f"W={W}, H={H}")
        msg = msgs[1]
        image = np.frombuffer(msg, dtype=np.uint8).reshape(H, W, -1).squeeze()
        image_ori = image.copy()

        # estimate pose
        pose_pred, pose_pred_homo, num_inliers = self.pose_estimator(image)
        if num_inliers > 20:
            print(f"OnePose inliers is enough: {num_inliers}.")
        else:
            print(f"OnePose inliers is not enough: {num_inliers}.")
            pose_pred_homo = np.zeros([4, 4], dtype=np.float32)
        # send pose to socket
        msg = pose_pred_homo.T.reshape(-1).astype(np.float32).tobytes()
        self.socket.send(msg, 0)
        print("Sending pose: ", pose_pred_homo)

    def run(self):
        while True:
            try:
                self.run_service()
            except KeyboardInterrupt:
                print("Closing server...")
                break


@hydra.main(config_path='cfg/', config_name='config.yaml')
def one_pose_server(cfg):
    # build one pose inference
    data_dirs = cfg.input.data_dirs
    sfm_model_dirs = cfg.input.sfm_model_dirs
    if isinstance(data_dirs, str) and isinstance(sfm_model_dirs, str):
        data_dirs = [data_dirs]
        sfm_model_dirs = [sfm_model_dirs]
    # get the first one
    sfm_data_dir = os.path.join(data_dirs[0].split(" ")[0], data_dirs[0].split(" ")[1])
    sfm_model_dir = sfm_model_dirs[0]
    # init one pose inference
    one_pose_inference = OnePoseInference(cfg, sfm_data_dir, sfm_model_dir) 

    pose_server = PoseServer(lambda image : one_pose_inference.inference(cfg, image), cfg.host, cfg.port)
    pose_server.run()


if __name__ == "__main__":
    one_pose_server()
    