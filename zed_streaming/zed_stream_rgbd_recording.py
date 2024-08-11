########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

"""
    Read a stream and display the left images using OpenCV
"""
import sys
import pyzed.sl as sl
import cv2
import argparse
import socket 
import numpy as np
import time
# import ogl_viewer.viewer as gl


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

def main():
    init_parameters = sl.InitParameters()
    init_parameters.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
    init_parameters.coordinate_units = sl.UNIT.MILLIMETER 

    init_parameters.sdk_verbose = 1
    init_parameters.set_from_stream(opt.ip_address.split(':')[0],int(opt.ip_address.split(':')[1]))
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
    depth_mat = sl.Mat(res.width, res.height, sl.MAT_TYPE.U16_C1, sl.MEM.CPU)


    camera_model = cam.get_camera_information().camera_model
    # Create OpenGL viewer
    # viewer = gl.GLViewer()
    # viewer.init(1, sys.argv, camera_model, res)
    
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name,on_mouse)
    print_camera_information(cam)
    print_help()
    switch_camera_settings()
    
    current_stack = []
    last = time.time()

    key = ''
    while key != 113:  # for 'q' key
        err = cam.grab(runtime) #Check that a new image is successfully acquired
        if err == sl.ERROR_CODE.SUCCESS:

            now = time.time()
            print("time_diff: ", now - last)
            cam.retrieve_image(mat, sl.VIEW.LEFT) #Retrieve left image
            cvImage = mat.get_data()
            cvImage = cvImage[:,:,:3]
            cam.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA,sl.MEM.CPU, res)
            err, point_cloud_value = point_cloud.get_value(960, 540)

            cam.retrieve_measure(depth_mat, sl.MEASURE.DEPTH_U16_MM)
            depth_value = depth_mat.get_data().astype(np.uint16)
            depth_value = depth_value.astype(float)
            point_cloud_xyz = point_cloud.get_data()[:, :, 0:3]

            print("point_cloud_xyz: ", point_cloud_xyz.shape, point_cloud_xyz[960, 540])
            print("bgr: ", cvImage.shape, cvImage[960, 540])
            print("depth_value: ", depth_value.shape, np.min(depth_value), np.max(depth_value))

            err, center_depth_value = depth_mat.get_value(960, 540)
            print("center_depth_value: ", center_depth_value)   
            # print("point_cloud_value: ", point_cloud_value)
            
            
            last = now
            

            current_data = {}
            current_data['bgr'] = cvImage
            current_data['xyz'] = point_cloud_xyz
            current_data['depth'] = depth_value

            print("depth_value: ", depth_value.shape, np.min(depth_value), np.max(depth_value))
            
            current_stack.append(current_data)
            if(len(current_stack) > 10):
                now = time.time()
                np.save( str(now), current_stack)
                break
            
            if (not selection_rect.is_empty() and selection_rect.is_contained(sl.Rect(0,0,cvImage.shape[1],cvImage.shape[0]))):
                cv2.rectangle(cvImage,(selection_rect.x,selection_rect.y),(selection_rect.width+selection_rect.x,selection_rect.height+selection_rect.y),(220, 180, 20), 2)
            cv2.imshow(win_name, cvImage)
        else:
            print("Error during capture : ", err)
            break
        key = cv2.waitKey(5)
        update_camera_settings(key, cam, runtime, mat)
    cv2.destroyAllWindows()
    cam.close()



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

  
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip_address', type=valid_ip_or_hostname, help='IP address or hostname of the sender. Should be in format a.b.c.d:p or hostname:p', required=True)
    opt = parser.parse_args()
    main()


