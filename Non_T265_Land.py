import cv2
import numpy as np
import transformations as tf
import math as m
import time
import argparse
import threading

from apscheduler.schedulers.background import BackgroundScheduler
from dronekit import connect, VehicleMode, LocationGlobalRelative, Command, LocationGlobal
from pymavlink import mavutil

from dt_apriltags import Detector



#######################################
# Parameters for FCU and MAVLink
#######################################

# Default configurations for connection to the FCU
connection_string_default = '/dev/serial0'
connection_baudrate_default = 921600
landing_target_msg_hz_default = 20

range_data = 0

# Timestamp (UNIX Epoch time or time since system boot)
current_time = 0

vehicle = None
is_landing_tag_detected = None

#######################################
# Parsing user' inputs
#######################################

parser = argparse.ArgumentParser(description='ArduPilot AprilTag Landing')
parser.add_argument('--camera_resolution', type=float,
                    help="Update the resoultion of captured image for landing pad detection. Higher resolution will slow down processing speed")
parser.add_argument('--connect',
                    help="Vehicle connection target string. If not specified, a default string will be used.")
parser.add_argument('--baudrate', type=float,
                    help="Vehicle connection baudrate. If not specified, a default value will be used.")
parser.add_argument('--landing_target_msg_hz', type=float,
                    help="Update frequency for LANDING_TARGET message. If not specified, a default value will be used.")


args = parser.parse_args()

camera_resolution = args.camera_resolution
connection_string = args.connect
connection_baudrate = args.baudrate
landing_target_msg_hz = args.landing_target_msg_hz



######################################
# Camera Resolution Setup
######################################

if not camera_resolution:
    camera_size=[640,480]
    print("INFO: Using Camera Resultion as", camera_size[1])
else:
    if camera_resolution == 720:
        camera_size=[1280,720]
    else:
        camera_size=[640,480]
    print("INFO: Using Camera Resultion as", camera_size[1])


if camera_size[1] == 720:
    camera_matrix   = np.loadtxt('cameraMatrix_720.txt', delimiter=',')
elif camera_size[1] == 480:
    camera_matrix   = np.loadtxt('cameraMatrix_480.txt', delimiter=',')
else:
    print('Error Loading Camera Matrix')
    exit()



if not connection_string:
    connection_string = connection_string_default
    print("INFO: Using default connection_string", connection_string)
else:
    print("INFO: Using connection_string", connection_string)

if not connection_baudrate:
    connection_baudrate = connection_baudrate_default
    print("INFO: Using default connection_baudrate", connection_baudrate)
else:
    print("INFO: Using connection_baudrate", connection_baudrate)

if not landing_target_msg_hz:
    landing_target_msg_hz = landing_target_msg_hz_default
    print("INFO: Using default landing_target_msg_hz", landing_target_msg_hz)
else:
    print("INFO: Using landing_target_msg_hz", landing_target_msg_hz)

#######################################
# Functions for AprilTag detection
#######################################

at_detector = Detector(searchpath=['apriltags'],
                       families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)
        
def send_land_message_v1():
    global current_time, H_camera_tag, is_landing_tag_detected

    if is_landing_tag_detected == True:
        x = H_camera_tag[0][3]
        y = H_camera_tag[1][3]
        z = H_camera_tag[2][3]

        x_offset_rad = m.atan(x / z)
        y_offset_rad = m.atan(y / z)
        distance = np.sqrt(x * x + y * y + z * z)

        msg = vehicle.message_factory.landing_target_encode(
            current_time,          # time target data was processed, as close to sensor capture as possible
            0,          # target num, not used
            mavutil.mavlink.MAV_FRAME_BODY_NED, # frame, not used
            x_offset_rad,          # X-axis angular offset, in radians
            y_offset_rad,          # Y-axis angular offset, in radians
            distance,          # distance, in meters
            0,          # Target x-axis size, in radians
            0,          # Target y-axis size, in radians
        )
        
        vehicle.send_mavlink(msg)
        vehicle.flush()

def vehicle_connect():
    global vehicle

    try:
        vehicle = connect(connection_string, wait_ready = True, baud = connection_baudrate, timeout=300)
    except KeyboardInterrupt:
        print("INFO: Exiting")
        sys.exit()
    except:
        print('Connection error! Retrying...')

    if vehicle == None:
        return False
    else:
        return True


#######################################
# Camera Setup
#######################################

camera_params = [camera_matrix[0][0], camera_matrix[1][1], camera_matrix[0][2], camera_matrix[1][2]]

vid = cv2.VideoCapture(0) 
        
vid.set(cv2.CAP_PROP_FRAME_WIDTH, camera_size[0])
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_size[1])

print("INFO: Connecting to vehicle.")
while (not vehicle_connect()):
    pass
print("INFO: Vehicle connected.")


sched = BackgroundScheduler()
sched.add_job(send_land_message_v1, 'interval', seconds = 1/landing_target_msg_hz_default)

sched.start()

try:
    while True:

        range_data = vehicle.rangefinder.distance
        print(range_data)
        
        if range_data < 0.5:
            tag_landing_id = 8
            tag_landing_size = 0.0495            # tag's border size, measured in meter
        else:
            tag_landing_id = 5
            tag_landing_size = 0.161            # tag's border size, measured in meter


        ret, frame = vid.read() 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        current_time = int(round(time.time() * 1000000))

        tags = at_detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_landing_size)
        
        if tags != []:
            for tag in tags:
                # Check for the tag that we want to land on
                if tag.tag_id == tag_landing_id:
                    is_landing_tag_detected = True
                    H_camera_tag = tf.euler_matrix(0, 0, 0, 'sxyz')
                    H_camera_tag[0][3] = tag.pose_t[0]
                    H_camera_tag[1][3] = tag.pose_t[1]
                    H_camera_tag[2][3] = tag.pose_t[2]
                    print("INFO: Detected landing tag", str(tag.tag_id), " relative to camera at x:", H_camera_tag[0][3], ", y:", H_camera_tag[1][3], ", z:", H_camera_tag[2][3])

        else:
            print("INFO: No tag detected")
            is_landing_tag_detected = False
            
except KeyboardInterrupt:
    sched.shutdown()
    vid.release()
    vehicle.close()
    print("INFO: KeyboardInterrupt has been caught. Cleaning up...")  