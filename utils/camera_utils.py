## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
from tqdm import tqdm
import cv2
import os
import shutil


class RealSenseCamera:
    def __init__(self) -> None:
        # Create a pipeline
        pipeline = rs.pipeline()

        # Create a config and configure the pipeline to stream
        #  different resolutions of color and depth streams
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        if device_product_line == 'L500':
            config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        profile = pipeline.start(config)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: " , depth_scale)

        # We will be removing the background of objects more than
        #  clipping_distance_in_meters meters away
        clipping_distance_in_meters = 1 #1 meter
        clipping_distance = clipping_distance_in_meters / depth_scale

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        self.pipeline = pipeline

    def get_frame(self):
        # Get frameset of color and depth
        frames = self.pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            return None, None

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        return depth_image, color_image
    
    def exit(self):
        self.pipeline.stop()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="", help="output dir for video")
    args = parser.parse_args()

    # Save frames into video
    if args.output_dir != "":
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        depth_image_path = os.path.join(args.output_dir, "depth")
        if os.path.exists(depth_image_path):
            shutil.rmtree(depth_image_path)
        os.makedirs(depth_image_path, exist_ok=True)
        color_image_path = os.path.join(args.output_dir, "color")
        if os.path.exists(color_image_path):
            shutil.rmtree(color_image_path)
        os.makedirs(color_image_path, exist_ok=True)
        depth_color_image_path = os.path.join(args.output_dir, "depth_color")
        if os.path.exists(depth_color_image_path):
            shutil.rmtree(depth_color_image_path)
        os.makedirs(depth_color_image_path, exist_ok=True)

        # Launch realsense camera
        camera = RealSenseCamera()
        import time
        time.sleep(2)  # Wait for camera to warm up

        # Image buffer
        color_image_list = []
        depth_image_list = []
        depth_color_image_list = []
        frame = 0
        while True:
            depth_image, color_image = camera.get_frame()
            if depth_image is None or color_image is None:
                continue
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            cv2.imshow("depth", depth_image)
            cv2.imshow("color", color_image)
            
            cv2.imwrite(os.path.join(color_image_path, "%d.png" % frame), color_image)
            cv2.imwrite(os.path.join(depth_image_path, "%d.png" % frame), depth_image)
            cv2.imwrite(os.path.join(depth_color_image_path, "%d.png" % frame), depth_colormap)

            frame += 1
            # Press esc or 'q' to close the image window
            k = cv2.waitKey(1)
            if k%256 == 27 or k%256 == ord('q'):
                print("Escape hit, closing...")
                break
        
        # Release camera and video
        camera.exit()