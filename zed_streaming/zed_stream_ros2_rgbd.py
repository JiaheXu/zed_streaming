#!/usr/bin/env python
"""
Inputs a Joy messages and produces controls
that match the control_interface
"""
import rclpy
from rclpy.node import Node

import time
import math
from geometry_msgs.msg import PointStamped, TwistStamped, Quaternion, Vector3
from std_msgs.msg import UInt8, Bool, String
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from std_msgs.msg import String, Float32, Int8, UInt8, Bool, UInt32MultiArray, Int32

import sys
import pyzed.sl as sl
import cv2
import argparse
import socket 
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from numpy.linalg import inv
from scipy.spatial.transform import Rotation

from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros import TransformBroadcaster
import open3d as o3d
from geometry_msgs.msg import PointStamped, TwistStamped, Quaternion, Vector3, TransformStamped
from std_msgs.msg import String, Float32, Int8, UInt8, Bool, UInt32MultiArray, Int32,Header

from sensor_msgs.msg import PointCloud2, PointField
# import sensor_msgs.point_cloud2 as pc2
import sensor_msgs_py.point_cloud2 as pc2
from ctypes import * # convert float to uint32
# The data structure of each point in ros PointCloud2: 16 bits = x + y + z + rgb
FIELDS_XYZ = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
]
FIELDS_XYZRGB = FIELDS_XYZ + \
    [PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)]
# from utils.ros2_o3d_utils import *
# Bit operations
BIT_MOVE_16 = 2**16
BIT_MOVE_8 = 2**8
convert_rgbUint32_to_tuple = lambda rgb_uint32: (
    (rgb_uint32 & 0x00ff0000)>>16, (rgb_uint32 & 0x0000ff00)>>8, (rgb_uint32 & 0x000000ff)
)
convert_rgbFloat_to_tuple = lambda rgb_float: convert_rgbUint32_to_tuple(
    int(cast(pointer(c_float(rgb_float)), POINTER(c_uint32)).contents.value)
)
convert_rgbaUint32_to_tuple = lambda rgb_uint32: (
    (rgb_uint32 & 0x00ff0000)>>16, (rgb_uint32 & 0x0000ff00)>>8, (rgb_uint32 & 0x000000ff), (rgb_uint32 & 0xff000000)>>24
)

camera_settings = sl.VIDEO_SETTINGS.BRIGHTNESS
str_camera_settings = "BRIGHTNESS"
step_camera_settings = 1
led_on = True 
selection_rect = sl.Rect()
select_in_progress = False
origin_rect = (-1,-1 )



def on_mouse(event,x,y,flags,param):
    global select_in_progress,selection_rect,origin_rect
    if event == cv2.EVENT_LBUTTONDOWN:
        origin_rect = (x, y)
        select_in_progress = True
    elif event == cv2.EVENT_LBUTTONUP:
        select_in_progress = False 
    elif event == cv2.EVENT_RBUTTONDOWN:
        select_in_progress = False 
        selection_rect = sl.Rect(0,0,0,0)
    
    if select_in_progress:
        selection_rect.x = min(x,origin_rect[0])
        selection_rect.y = min(y,origin_rect[1])
        selection_rect.width = abs(x-origin_rect[0])+1
        selection_rect.height = abs(y-origin_rect[1])+1
def print_camera_information(cam):
    cam_info = cam.get_camera_information()
    print("ZED Model                 : {0}".format(cam_info.camera_model))
    print("ZED Serial Number         : {0}".format(cam_info.serial_number))
    print("ZED Camera Firmware       : {0}/{1}".format(cam_info.camera_configuration.firmware_version,cam_info.sensors_configuration.firmware_version))
    print("ZED Camera Resolution     : {0}x{1}".format(round(cam_info.camera_configuration.resolution.width, 2), cam.get_camera_information().camera_configuration.resolution.height))
    print("ZED Camera FPS            : {0}".format(int(cam_info.camera_configuration.fps)))


def print_help():
    print("\n\nCamera controls hotkeys:")
    print("* Increase camera settings value:  '+'")
    print("* Decrease camera settings value:  '-'")
    print("* Toggle camera settings:          's'")
    print("* Toggle camera LED:               'l' (lower L)")
    print("* Reset all parameters:            'r'")
    print("* Reset exposure ROI to full image 'f'")
    print("* Use mouse to select an image area to apply exposure (press 'a')")
    print("* Exit :                           'q'\n")

#Update camera setting on key press
def update_camera_settings(key, cam, runtime, mat):
    global led_on
    if key == 115:  # for 's' key
        # Switch camera settings
        switch_camera_settings()
    elif key == 43:  # for '+' key
        # Increase camera settings value.
        current_value = cam.get_camera_settings(camera_settings)[1]
        cam.set_camera_settings(camera_settings, current_value + step_camera_settings)
        print(str_camera_settings + ": " + str(current_value + step_camera_settings))
    elif key == 45:  # for '-' key
        # Decrease camera settings value.
        current_value = cam.get_camera_settings(camera_settings)[1]
        if current_value >= 1:
            cam.set_camera_settings(camera_settings, current_value - step_camera_settings)
            print(str_camera_settings + ": " + str(current_value - step_camera_settings))
    elif key == 114:  # for 'r' key
        # Reset all camera settings to default.
        cam.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS, -1)
        cam.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST, -1)
        cam.set_camera_settings(sl.VIDEO_SETTINGS.HUE, -1)
        cam.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, -1)
        cam.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, -1)
        cam.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, -1)
        cam.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, -1)
        cam.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE, -1)
        print("[Sample] Reset all settings to default")
    elif key == 108: # for 'l' key
        # Turn on or off camera LED.
        led_on = not led_on
        cam.set_camera_settings(sl.VIDEO_SETTINGS.LED_STATUS, led_on)
    elif key == 97 : # for 'a' key 
        # Set exposure region of interest (ROI) on a target area.
        print("[Sample] set AEC_AGC_ROI on target [",selection_rect.x,",",selection_rect.y,",",selection_rect.width,",",selection_rect.height,"]")
        cam.set_camera_settings_roi(sl.VIDEO_SETTINGS.AEC_AGC_ROI,selection_rect,sl.SIDE.BOTH)
    elif key == 102: #for 'f' key 
        # Reset exposure ROI to full resolution.
        print("[Sample] reset AEC_AGC_ROI to full res")
        cam.set_camera_settings_roi(sl.VIDEO_SETTINGS.AEC_AGC_ROI,selection_rect,sl.SIDE.BOTH,True)

# Function to switch between different camera settings (brightness, contrast, etc.).
def switch_camera_settings():
    global camera_settings
    global str_camera_settings
    if camera_settings == sl.VIDEO_SETTINGS.BRIGHTNESS:
        camera_settings = sl.VIDEO_SETTINGS.CONTRAST
        str_camera_settings = "Contrast"
        print("[Sample] Switch to camera settings: CONTRAST")
    elif camera_settings == sl.VIDEO_SETTINGS.CONTRAST:
        camera_settings = sl.VIDEO_SETTINGS.HUE
        str_camera_settings = "Hue"
        print("[Sample] Switch to camera settings: HUE")
    elif camera_settings == sl.VIDEO_SETTINGS.HUE:
        camera_settings = sl.VIDEO_SETTINGS.SATURATION
        str_camera_settings = "Saturation"
        print("[Sample] Switch to camera settings: SATURATION")
    elif camera_settings == sl.VIDEO_SETTINGS.SATURATION:
        camera_settings = sl.VIDEO_SETTINGS.SHARPNESS
        str_camera_settings = "Sharpness"
        print("[Sample] Switch to camera settings: Sharpness")
    elif camera_settings == sl.VIDEO_SETTINGS.SHARPNESS:
        camera_settings = sl.VIDEO_SETTINGS.GAIN
        str_camera_settings = "Gain"
        print("[Sample] Switch to camera settings: GAIN")
    elif camera_settings == sl.VIDEO_SETTINGS.GAIN:
        camera_settings = sl.VIDEO_SETTINGS.EXPOSURE
        str_camera_settings = "Exposure"
        print("[Sample] Switch to camera settings: EXPOSURE")
    elif camera_settings == sl.VIDEO_SETTINGS.EXPOSURE:
        camera_settings = sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE
        str_camera_settings = "White Balance"
        print("[Sample] Switch to camera settings: WHITEBALANCE")
    elif camera_settings == sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE:
        camera_settings = sl.VIDEO_SETTINGS.BRIGHTNESS
        str_camera_settings = "Brightness"
        print("[Sample] Switch to camera settings: BRIGHTNESS")

def valid_ip_or_hostname(ip_or_hostname):
    try:
        host, port = ip_or_hostname.split(':')
        socket.inet_aton(host)  # Vérifier si c'est une adresse IP valide
        port = int(port)
        return f"{host}:{port}"
    except (socket.error, ValueError):
        raise argparse.ArgumentTypeError("Invalid IP address or hostname format. Use format a.b.c.d:p or hostname:p")

  
class zed_streamer(Node):
    
    def __init__(self):
        super().__init__('zed_stream_to_ros_node')

        self.declare_parameter('ip_address', '192.168.0.34:3000')
        self.ip_address = self.get_parameter('ip_address').get_parameter_value().string_value
        self.bridge = CvBridge() 
        # self.master_cam_t = TransformStamped()

        # self.master_cam_t.header.frame_id = 'world'
        # self.master_cam_t.child_frame_id = "master_camera"
        # self.master_cam_t.transform.translation.x = -0.1393031
        # self.master_cam_t.transform.translation.y = 0.0539
        # self.master_cam_t.transform.translation.z = 0.43911375

        # self.master_cam_t.transform.rotation.x = -0.61860094
        # self.master_cam_t.transform.rotation.y = 0.66385477
        # self.master_cam_t.transform.rotation.z = -0.31162288
        # self.master_cam_t.transform.rotation.w = 0.2819945

        # Todo: use yaml files
        self.cam_extrinsic = self.get_transform( [-0.13913296, 0.053, 0.43643044, -0.63127772, 0.64917582, -0.31329509, 0.28619116])
        self.o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(1920, 1080, 734.1779174804688, 734.1779174804688, 993.6226806640625, 551.8895874023438)

        self.resized_image_size = (256,256)
        self.original_image_size = (1080, 1920) #(h,)
        fxfy = 256.0
        self.resized_intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(256, 256, fxfy, fxfy, 128.0, 128.0)
        self.resized_intrinsic_np = np.array([
            [fxfy, 0., 128.0],
            [0. ,fxfy,  128.0],
            [0., 0., 1.0]
        ])

        self.image_pub = self.create_publisher(Image, "left_image", 1)
        # self.image_pub = self.create_publisher(String, "left_image", 1)
        self.depth_pub = self.create_publisher(Image, "depth", 1)
        
        self.pcd_publisher = self.create_publisher(PointCloud2, "rgb_pcd", 1)

    def get_transform(self, transf_7D):
        trans = transf_7D[0:3]
        quat = transf_7D[3:7]
        t = np.eye(4)
        t[:3, :3] = Rotation.from_quat( quat ).as_matrix()
        t[:3, 3] = trans
        return t

    # Convert the datatype of point cloud from Open3D to ROS PointCloud2 (XYZRGB only)
    def convertCloudFromOpen3dToRos(self, open3d_cloud, frame_id="world"):
        # Set "header"

        ros_time = self.get_clock().now()
        header = Header()
        header.stamp = ros_time.to_msg()
        header.frame_id = frame_id

        # Set "fields" and "cloud_data"
        points=np.asarray(open3d_cloud.points)
        if not open3d_cloud.colors: # XYZ only
            fields=FIELDS_XYZ
            cloud_data=points
        else: # XYZ + RGB
            fields=FIELDS_XYZRGB
            # -- Change rgb color from "three float" to "one 24-byte int"
            # 0x00FFFFFF is white, 0x00000000 is black.
            colors = np.floor(np.asarray(open3d_cloud.colors)*255) # nx3 matrix
            colors = colors[:,0] * BIT_MOVE_16 +colors[:,1] * BIT_MOVE_8 + colors[:,2]  
            cloud_data=np.c_[points, colors]
        
        # create ros_cloud
        # fields=FIELDS_XYZ
        # cloud_data=points
        # print("fields: ", fields)
        return pc2.create_cloud(header, fields, cloud_data)

    def image_process(self, bgr, depth, intrinsic_np, original_img_size, resized_intrinsic_np, resized_img_size):
        
        cx = intrinsic_np[0,2]
        cy = intrinsic_np[1,2]

        fx_factor = resized_intrinsic_np[0,0] / intrinsic_np[0,0]
        fy_factor = resized_intrinsic_np[1,1] / intrinsic_np[1,1]

        raw_fx = resized_intrinsic_np[0,0] * intrinsic_np[0,0] / resized_intrinsic_np[0,0]
        raw_fy = resized_intrinsic_np[1,1] * intrinsic_np[1,1] / resized_intrinsic_np[1,1]
        raw_cx = resized_intrinsic_np[0,2] * intrinsic_np[0,0] / resized_intrinsic_np[0,0]
        raw_cy = resized_intrinsic_np[1,2] * intrinsic_np[1,1] / resized_intrinsic_np[1,1]

        width = resized_img_size[0] * intrinsic_np[0,0] / resized_intrinsic_np[0,0]
        height = resized_img_size[0] * intrinsic_np[1,1] / resized_intrinsic_np[1,1]
        
        half_width = int( width / 2.0 )
        half_height = int( height / 2.0 )

        cropped_bgr = bgr[round(cy-half_height) : round(cy + half_height), round(cx - half_width) : round(cx + half_width), :]
        cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
        processed_rgb = cv2.resize(cropped_rgb, resized_img_size)

        cropped_depth = depth[round(cy-half_height) : round(cy + half_height), round(cx - half_width) : round(cx + half_width)]
        processed_depth = cv2.resize(cropped_depth, resized_img_size, interpolation =cv2.INTER_NEAREST)

        return processed_rgb, processed_depth

    def run(self):
        init_parameters = sl.InitParameters()
        init_parameters.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
        #init_parameters.depth_mode = sl.DEPTH_MODE.NEURAL
        init_parameters.coordinate_units = sl.UNIT.MILLIMETER
        # init_parameters.depth_maximum_distance = 2000 # 2m
        init_parameters.depth_minimum_distance = 300 # 0.3m
        init_parameters.sdk_verbose = 1
        init_parameters.set_from_stream( self.ip_address.split(':')[0],int(  self.ip_address.split(':')[1]))
        cam = sl.Camera()
        status = cam.open(init_parameters)
        if status != sl.ERROR_CODE.SUCCESS:
            print("Camera Open : "+repr(status)+". Exit program.")
            exit()
        runtime = sl.RuntimeParameters()
        win_name = "Camera Remote Control"
        
        mat = sl.Mat()
        depth_mat = sl.Mat(1920, 1080, sl.MAT_TYPE.U16_C1, sl.MEM.CPU)

        
        # cv2.namedWindow(win_name)
        # cv2.setMouseCallback(win_name,on_mouse)
        print_camera_information(cam)
        print_help()
        switch_camera_settings()

        key = ''
        while True:
            if key == 113:  # for 'q' key
                break
            err = cam.grab(runtime) #Check that a new image is successfully acquired
            if err == sl.ERROR_CODE.SUCCESS:
                cam.retrieve_image(mat, sl.VIEW.LEFT) #Retrieve left image
                timestamp = cam.get_timestamp(sl.TIME_REFERENCE.IMAGE)
                cvImage = mat.get_data()
                cvImage = cvImage[:,:,:3]
                img_msg = self.bridge.cv2_to_imgmsg(cvImage, encoding="bgr8")
                
                
                # cam.retrieve_measure(depth_mat, sl.MEASURE.DEPTH) #Retrieve depth image
                cam.retrieve_measure(depth_mat, sl.MEASURE.DEPTH_U16_MM)
                depth_value = depth_mat.get_data().astype(np.uint16)
                # print("depth_value: ", depth_value[540, 960])
                depth_msg = self.bridge.cv2_to_imgmsg(depth_value, encoding = "mono16")

                img_msg.header.stamp = rclpy.time.Time(seconds=timestamp.get_seconds(), nanoseconds=timestamp.get_nanoseconds()%1000000000).to_msg()
                depth_msg.header.stamp = rclpy.time.Time(seconds=timestamp.get_seconds(), nanoseconds=timestamp.get_nanoseconds()%1000000000).to_msg()

                # print("ros2 time: ", img_msg.header.stamp)
                self.image_pub.publish(img_msg)
                self.depth_pub.publish(depth_msg)

                # self.tf_broadcaster.sendTransform(self.master_cam_t)

                ###############################################
                # point cloud
                ###############################################
                bgr_np = np.array(self.bridge.imgmsg_to_cv2(img_msg))[:,:,:3]
                depth_np = np.array(self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="mono16"))

                rgb, depth = self.image_process(bgr_np, 
                                                depth_np, 
                                                self.o3d_intrinsic.intrinsic_matrix, 
                                                self.original_image_size, 
                                                self.resized_intrinsic_o3d.intrinsic_matrix,
                                                self.resized_image_size 
                                                )
                im_color = o3d.geometry.Image(rgb)
                im_depth = o3d.geometry.Image(depth)
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    im_color, im_depth, depth_scale=1000, depth_trunc=2000, convert_rgb_to_intensity=False)
                original_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                        rgbd,
                        self.resized_intrinsic_o3d
                    )
                original_pcd = original_pcd.transform( self.cam_extrinsic )
                pcd_msg = self.convertCloudFromOpen3dToRos(original_pcd, frame_id="world")
                self.pcd_publisher.publish(pcd_msg)

            else:
                print("Error during capture : ", err)
                break
            # key = cv2.waitKey(5)
            update_camera_settings(key, cam, runtime, mat)

         

        # cv2.destroyAllWindows()

        cam.close()
        


def main(args=None):

    rclpy.init(args=args)
    node = zed_streamer()
    node.run()
 
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
