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
        self.connected = False

    def wait_for_connection(self):
        print("No connection. Waiting for connection...")
        self.connection, self.client_address = self.sock.accept()
        print("Connection from: ", self.client_address)
        self.connected = True

    def run_service(self):
        # receive image size
        print("Receiving image size...")
        data = self.connection.recv(4)
        if data:
            image_size = int.from_bytes(data, byteorder='little')
            # receive image
            image_data = b''
            while len(image_data) < image_size:
                image_data += self.connection.recv(1024)
            image = np.frombuffer(image_data, dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            # estimate pose
            pose_pred, pose_pred_homo, num_inliers = self.pose_estimator(image)
            if num_inliers > 20:
                print(f"OnePose inliers is enough: {num_inliers}.")
            else:
                print(f"OnePose inliers is not enough: {num_inliers}.")
                pose_pred_homo = np.zeros([4, 4], dtype=np.float32)
            # send pose to socket
            data = bytes(pose_to_string(pose_pred_homo), 'utf-8')
            print("Sending pose: ", data)
            self.connection.sendall(data)
            # Set connected to False to wait for next connection
            self.connected = False

    def close(self):
        self.connection.close()
        self.sock.close()

    def run(self):
        while True:
            try:
                if not self.connected:
                    self.wait_for_connection()
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
    pose_server.close()


if __name__ == "__main__":
    one_pose_server()
    