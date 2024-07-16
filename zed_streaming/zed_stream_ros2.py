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
from sensor_msgs.msg import Joy
from airlab_msgs.msg import AIRLABModes
from straps_msgs.msg import CmdPayloadTrackTargetRequest
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from std_msgs.msg import String, Float32, Int8, UInt8, Bool, UInt32MultiArray, Int32

import sys
import pyzed.sl as sl
import cv2
import argparse
import socket 

from sensor_msgs.msg import Image
from cv_bridge import CvBridge


camera_settings = sl.VIDEO_SETTINGS.BRIGHTNESS
str_camera_settings = "BRIGHTNESS"
step_camera_settings = 1
led_on = True 
selection_rect = sl.Rect()
select_in_progress = False
origin_rect = (-1,-1 )

bridge = CvBridge() 

class zed_streamer(Node):
    
    def __init__(self):
        super().__init__('zed_stream_to_ros_node')
        
 

        self.declare_parameter('ip_address', '192.168.0.34:3000')
        self.ip_address = self.get_parameter('ip_address').get_parameter_value().string_value


        self.image_pub = self.create_publisher(Image, "left_image", 1)
        self.depth_pub = self.create_publisher(Image, "depth", 1)
        
        # timer_period = 1.0
        # self.timer = self.create_timer(timer_period, self.publish_inspection)

        init_parameters = sl.InitParameters()
        init_parameters.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
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
        cv2.namedWindow(win_name)
        cv2.setMouseCallback(win_name,on_mouse)
        print_camera_information(cam)
        print_help()
        switch_camera_settings()

        key = ''
        while not rospy.is_shutdown():
            if key == 113:  # for 'q' key
                break
            err = cam.grab(runtime) #Check that a new image is successfully acquired
            if err == sl.ERROR_CODE.SUCCESS:
                cam.retrieve_image(mat, sl.VIEW.LEFT) #Retrieve left image
                timestamp = cam.get_timestamp(sl.TIME_REFERENCE.IMAGE)
                print("nano time: ", timestamp.get_nanoseconds())
                print("timestamp: ", timestamp.get_seconds(), timestamp.get_nanoseconds()%1000000000)
                cvImage = mat.get_data()
                cvImage = cvImage[:,:,:3]
                # filename = "./cam0/{}.jpg".format( str(timestamp.get_milliseconds()) )
                # cv2.imwrite(filename, cvImage)
                
                img_msg = bridge.cv2_to_imgmsg(cvImage, encoding="rgb8")
                # img_msg.header.stamp=rospy.Time.now()
                # img_msg.header.stamp = timestamp.get_nanoseconds()
                
                
                # img_msg.header.stamp = rospy.Time(timestamp.get_seconds(), timestamp.get_nanoseconds()%1000000000)
                img_msg.header.stamp = rclpy.time.Time(timestamp.get_nanoseconds())
                print("ros2 time: ", img_msg.header.stamp)
                image_pub.publish(img_msg)
                if (not selection_rect.is_empty() and selection_rect.is_contained(sl.Rect(0,0,cvImage.shape[1],cvImage.shape[0]))):
                    cv2.rectangle(cvImage,(selection_rect.x,selection_rect.y),(selection_rect.width+selection_rect.x,selection_rect.height+selection_rect.y),(220, 180, 20), 2)
                cv2.imshow(win_name, cvImage)
            else:
                print("Error during capture : ", err)
                break
            key = cv2.waitKey(5)
            update_camera_settings(key, cam, runtime, mat)

            rate.sleep()

        cv2.destroyAllWindows()

        cam.close()
        


def main():

    rclpy.init(args=args)
    node = zed_streamer()

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
