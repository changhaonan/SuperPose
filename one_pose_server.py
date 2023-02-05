import socket
import numpy as np
import cv2
import os
import cv2
import numpy as np
import hydra
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
        self.host = host
        self.port = port
        self.pose_estimator = pose_estimator
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.host, self.port))
        self.sock.listen(1)
        print("Waiting for connection...")
        self.connection, self.client_address = self.sock.accept()
        print("Connection from: ", self.client_address)

    def send_pose(self, pose):
        data = self.connection.recv(1024)
        if data:
            data = bytes(pose_to_string(pose), 'utf-8')
            print("Sending pose: ", data)
            self.connection.sendall(data)

    def close(self):
        self.connection.close()
        self.sock.close()

    def run(self):
        while True:
            try:
                pose = np.eye(4)
                self.send_pose(pose)
            except KeyboardInterrupt:
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

    pose_server = PoseServer(one_pose_inference, cfg.host, cfg.port)
    pose_server.run()
    pose_server.close()


if __name__ == "__main__":
    one_pose_server()
    