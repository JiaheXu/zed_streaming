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

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs
import numpy as np


camera_settings = sl.VIDEO_SETTINGS.BRIGHTNESS
str_camera_settings = "BRIGHTNESS"
step_camera_settings = 1
led_on = True 
selection_rect = sl.Rect()
select_in_progress = False
origin_rect = (-1,-1 )

bridge = CvBridge() 

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

        self.image_pub = self.create_publisher(Image, "left_image", 1)
        # self.image_pub = self.create_publisher(String, "left_image", 1)
        self.depth_pub = self.create_publisher(Image, "depth", 1)

    def point_cloud(self, points):
        """ Creates a point cloud message.
        Args:
            points: Nx6 array of xyz positions (m) and rgba colors (0..1)
            parent_frame: frame in which the point cloud is defined
        Returns:
            sensor_msgs/PointCloud2 message
        """
        ros_dtype = sensor_msgs.PointField.FLOAT32
        dtype = np.float32
        itemsize = np.dtype(dtype).itemsize

        data = points.astype(dtype).tobytes()

        fields = [sensor_msgs.PointField(
            name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
            for i, n in enumerate('xyz_color')]

        # header = std_msgs.Header(frame_id=parent_frame, stamp=rospy.Time.now())
        # img_msg.header.stamp = rclpy.time.Time(seconds=timestamp.get_seconds(), nanoseconds=timestamp.get_nanoseconds()%1000000000).to_msg()
        header = std_msgs.Header()
        return sensor_msgs.PointCloud2(
            header=header,
            height=1,
            width=points.shape[0],
            is_dense=False,
            is_bigendian=False,
            fields=fields,
            point_step=(itemsize * 4),
            row_step=(itemsize * 4 * points.shape[0]),
            data=data
        )
    
    def run(self):
        init_parameters = sl.InitParameters()
        init_parameters.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
        # init_parameters.coordinate_units = sl.UNIT.MILLIMETER
        init_parameters.coordinate_units = sl.UNIT.METER
        
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
        res = sl.Resolution()
        point_cloud = sl.Mat(res.width, res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
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
                img_msg = bridge.cv2_to_imgmsg(cvImage, encoding="bgr8")
                
                cam.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA,sl.MEM.CPU, res)
                point_cloud_xyz_color = point_cloud.get_data()


                print("point_cloud_xyz: ", point_cloud_xyz_color.shape, point_cloud_xyz_color[960, 540])
                print("bgr: ", cvImage.shape, cvImage[960, 540])
                
                img_msg.header.stamp = rclpy.time.Time(seconds=timestamp.get_seconds(), nanoseconds=timestamp.get_nanoseconds()%1000000000).to_msg()
                print("ros2 time: ", img_msg.header.stamp)
                # self.image_pub.publish(img_msg)

                self.point_cloud(point_cloud_xyz_color)

                if (not selection_rect.is_empty() and selection_rect.is_contained(sl.Rect(0,0,cvImage.shape[1],cvImage.shape[0]))):
                    cv2.rectangle(cvImage,(selection_rect.x,selection_rect.y),(selection_rect.width+selection_rect.x,selection_rect.height+selection_rect.y),(220, 180, 20), 2)
                # cv2.imshow(win_name, cvImage)
            else:
                print("Error during capture : ", err)
                break
            key = cv2.waitKey(5)
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
