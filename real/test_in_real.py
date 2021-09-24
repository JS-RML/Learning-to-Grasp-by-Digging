# -*- coding: utf-8 -*-
import pyrealsense2 as rs
import numpy as np
import math
from scipy import stats
from matplotlib import pyplot as plt
import urx
import cv2
from gripper140 import Robotiq_Two_Finger_Gripper
import time
import socket
from PIL import Image
from models import Dignet
import torch
import network as network_test_real
import os
import random
import math3d as m3d
import serial
import signal
from cam_im import get_pointcloud
import time

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

serialPort = "/dev/ttyACM0"
baudRate = 57600
ser = serial.Serial(serialPort, baudRate, timeout=0.5)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('test network device', device)

def myHandler(signum, frame):
    pass


def goto_c_home():
    "camera home"
    home_position = [125.02,-82.26,-109.19,-77.83,89.77,-54.98]
    #home_position = [124.46,-82.84,-108.69,-77.76,89.77,-55.54] 
    Hong_joint0 = math.radians(home_position[0])
    Hong_joint1 = math.radians(home_position[1])
    Hong_joint2 = math.radians(home_position[2])
    Hong_joint3 = math.radians(home_position[3])
    Hong_joint4 = math.radians(home_position[4])
    Hong_joint5 = math.radians(home_position[5])

    rob.movej((Hong_joint0, Hong_joint1, Hong_joint2, Hong_joint3, Hong_joint4, Hong_joint5), 0.3, 0.5)

def goto_rhome(home_position):

    Hong_joint0 = math.radians(home_position[0])
    Hong_joint1 = math.radians(home_position[1])
    Hong_joint2 = math.radians(home_position[2])
    Hong_joint3 = math.radians(home_position[3])
    Hong_joint4 = math.radians(home_position[4])
    Hong_joint5 = math.radians(home_position[5])
    rob.movej((Hong_joint0, Hong_joint1, Hong_joint2, Hong_joint3, Hong_joint4, Hong_joint5), 1, 2)

def gp_control(pos, speed, force, delay_time = 0.7):
        robotiqgrip.gripper_action_full(pos, speed, force)
        time.sleep(delay_time)

def resetFT300Sensor(tcp_host_ip):
    global serialFT300Sensor
    HOST = tcp_host_ip
    PORT = 63351
    serialFT300Sensor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serialFT300Sensor.connect((HOST, PORT))

def getFT300SensorData():
    global serialFT300Sensor
    while True:
        data = str(serialFT300Sensor.recv(1024),"utf-8").replace("(","").replace(")","").split(",")
        try:
            data = [float(x) for x in data]
            if len(data) == 6:
                break
        except:
            pass
    return data

def fg_length_control(d):
    global ser
    signal.signal(signal.SIGALRM, myHandler)
    signal.setitimer(signal.ITIMER_REAL, 0.001)
    ser.write(str.encode(d))
    signal.setitimer(signal.ITIMER_REAL, 0)
    time.sleep(.5)

def aper_to_gp_angle(aperture):
    desire_x = aperture #aperture in cm
    x = np.array([12.752, 12.436, 11.865, 11.283, 10.769, 10.212, 9.634, 9.047, 8.443, 7.820, 7.219, 6.601, 5.962, 5.313, 4.684, 4.014, 3.366, 2.759, 2.138, 1.483, 0.914, 0.319, 0])
    y = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220])
    res = stats.linregress(x, y)
    pred_y = res.intercept + res.slope*desire_x

    return int(pred_y)

def fg_diff_to_extf_angle(fg_diff):
    desire_x = 40-fg_diff # in mm
    x = np.array([0, 4.21, 5.43, 7.19, 9.31, 11.49, 13.15, 14.55, 16.28, 17.74, 18.61, 20.17, 21.74, 23.35, 25.08, 27.22, 29.1, 31.06, 32.88])
    y = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180])
    res = stats.linregress(x, y)
    pred_y = res.intercept + res.slope*desire_x

    return pred_y

def get_prediction_vis(predictions, color_heightmap, best_ind_in_all, yaw):

    canvas = None
    #num_rotations = predictions.shape[0]
    prediction_vis = predictions[best_ind_in_all[0],:,:]

    prediction_vis = Image.fromarray(prediction_vis)
    prediction_vis = np.array(prediction_vis.rotate(angle=-yaw, fillcolor=255))
    background_image = color_heightmap

    # Normalize prediction to (0-1)
    #prediction_vis = (prediction_vis-np.ones((prediction_vis.shape[0],prediction_vis.shape[1]))*np.min(prediction_vis))/(np.max(prediction_vis)-np.min(prediction_vis))
    prediction_vis = cv2.applyColorMap((prediction_vis*255).astype(np.uint8), cv2.COLORMAP_JET)
#    background_image = Image.fromarray(color_heightmap)
#    background_image = np.array(background_image.rotate(angle=yaw, fillcolor=(255,255,255)))
    prediction_vis_ = (0.6*cv2.cvtColor(background_image, cv2.COLOR_RGB2BGR) + 0.4*prediction_vis[:,:,[2,1,0]]).astype(np.uint8)
#    cv2.circle(prediction_vis, (int(best_ind_in_all[2]), int(best_ind_in_all[1])), 7, (255,0,0), 1)
    return prediction_vis_, background_image[:,:,[2,1,0]], prediction_vis[:,:,[2,1,0]]


#%% go camera home
rob = urx.Robot("192.168.1.102")
robotiqgrip = Robotiq_Two_Finger_Gripper(rob)
serialFT300Sensor = 0
goto_rhome([123.55,-81.70,-98.23,-89.59,89.67,-55.32])
fg_length_control('0')
time.sleep(.5)
gp_control(169,100,1)
rob.set_tcp((0.0205, 0.000, 0.3635, 0, 0, 0))
time.sleep(.5)
#print(rob.get_pose())
resetFT300Sensor("192.168.1.102")
goto_c_home()

#model_path = '../pretrain.ckpt'
model_path = './round7.ckpt'
model = Dignet(num_input_channels=3)
state_dict = {k.replace('auto_encoder.', ''): v for k, v in torch.load(model_path,map_location=device)['state_dict'].items()}
model.load_state_dict(state_dict)
model.to(device)

round = 1

while(True):
    #input('next round?')

    goto_rhome([125.82,-79.64,-105.07,-84.58,89.80,-54.12])
    _, color = get_pointcloud()

    goto_c_home()
    time.sleep(1)
    verts, _ = get_pointcloud()
    cx = verts[:,:,0]
    cy = verts[:,:,1]
    cz = verts[:,:,2]# + np.ones((cx.shape[0], cx.shape[1]))*0.01
    cz[np.where(cz>0.46)]=0.46

    '''
    origin = cz.copy()
    Eye_position = np.max(origin)
    origin = np.floor(1/Eye_position*origin*255).astype(np.int)

    origin = np.expand_dims(origin, axis=2)
    origin = np.concatenate((origin, origin, origin), axis=-1).astype(np.uint8)
    origin = Image.fromarray(origin)
    '''

    "process depth_image for network input"
    depth_image = cz.copy()
    depth_image[np.where(depth_image>0.3)]=0.46
    Eye_position = np.max(cz)
    print('Eye_position',Eye_position)
    depth_image = np.floor(depth_image*255/Eye_position).astype(np.uint8)
    prob_mask = np.ones(depth_image.shape)
    #plt.imshow(depth_image)
    depth_image[np.where(cz>0.3)]=255
#    depth_image[np.where(cz<0.25)]=255

    mask_temp = np.zeros((240,240))
    cv2.circle(mask_temp, (120,120),130, 255, -1)
    prob_mask = np.zeros((240,240))
    prob_mask[np.where(mask_temp==255)] = 1
    #prob_mask = 1

#    f = plt.figure(1)
#    f.add_subplot(1,2,1)
#    plt.imshow(depth_image)
#    f.add_subplot(1,2,2)
#    plt.imshow(depth_image*prob_mask)
#    plt.show()
#    depth_plot = np.hstack((depth_image,depth_image*prob_mask))
#    cv2.imshow('Depth', depth_plot)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()


    #plt.imshow(depth_image)10 180
    #plt.show()
    depth_image = np.expand_dims(depth_image, axis=2)
    depth_image= np.concatenate((depth_image, depth_image, depth_image), axis=-1).astype(np.uint8)
    #plt.imshow(depth_image)
    depth_image = depth_image.astype(np.uint8)
    #plt.imshow(depth_image)
    #plt.imshow(depth_image)
    #plt.show()

    predict, pixel_row,pixel_col, yaw, pitch, roll, aperture, fl, out_index, out_row, out_col, good_prob = network_test_real.test4(model,depth_image,prob_mask, 1)

    #aperture = 2.5
    #fl =
    #pixel_row =
    #pixel_col =
    grip_param = [roll, pitch, yaw, aperture, fl]
#    pixel_row = 121
#    pixel_col = 117
#    pixel_row = pixel_row[0]
#    pixel_col = pixel_col[0]
    ## for temp show
#    in_mask = np.array(np.where(center_mask==1))
#    in_mask_ind = random.randint(0,in_mask.shape[1]-1)
#    in_mask = in_mask[:,in_mask_ind]
#    pixel_row = in_mask[0]#pixel_row[0]
#    pixel_col = in_mask[1]#pixel_col[0]
#    yaw = 270
#    aperture = random.randint(0,1) * 10 + 25 #mm    dg_pos_in_base = np.matmul(baseTcam, np.array([cx[pixel_row,pixel_col],cy[pixel_row,pixel_col],cz[pixel_row,pixel_col],1]))

#    pitch = random.randint(0,2) * 10
#    roll = random.randint(0,2) * 10 - 10
#    fg_dl = random.randint(0,3) * 10 #mm

    #print("selected yaw, pitch, roll, aperture", yaw, pitch, roll, aperture, pixel_row, pixel_col)
    plot_depth = depth_image.copy()
    #plot_color = color.copy()
    color_resize_plot = Image.fromarray(color.copy())
    color_resize_plot = np.array(color_resize_plot.resize((240,240)))
    #cv2.circle(plot_depth, (pixel_col,pixel_row), 5, (255,0,0), -1)
    #cv2.circle(plot_color, (pixel_col,pixel_row), 5, (255,0,0), -1)
    canvas, bg_save, pd_save = get_prediction_vis(predict, color_resize_plot, [out_index, out_row, out_col], yaw)
    #plt.imshow(canvas)
    #plt.show()
    cv2.imwrite('./fig6/lastfig1/prediction_vis_'+str(round)+'.png',canvas[:,:,[2,1,0]])
    cv2.imwrite('./fig6/lastfig1/color_'+str(round)+'.png', color_resize_plot)
    np.save('./fig6/lastfig1/depth_'+str(round)+'.npy', depth_image)
    np.save('./fig6/lastfig1/origin_depth_'+str(round)+'.npy', cz)
    np.save('./fig6/lastfig1/grip_param_'+str(round)+'.npy', grip_param)
    np.save('./fig6/lastfig1/save_bg_'+str(round)+'.npy', bg_save)
    np.save('./fig6/lastfig1/save_pd_'+str(round)+'.npy', pd_save)
    np.save('./fig6/lastfig1/prob_'+str(round)+'.npy', good_prob)

    #input("check visual!!!!")

    #f = plt.figure(1)
    #f.add_subplot(2,2,1)
    #plt.imshow(predict[:,:,[2,1,0]])
    #f.add_subplot(2,2,2)
    #plt.imshow(plot_depth)
    #f.add_subplot(2,2,3)
    #plt.imshow(plot_color[:,:,[2,1,0]])
    #plt.show()

#    out_plot = np.hstack((predict,plot_depth,plot_color))
#    cv2.imshow('Out', out_plot)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()

    #print(cx[246,318-80])
    #print(cy[246,318-80])50
    #print(cz[246,318-80])

    rob.set_tcp((0, 0, 0, 0, 0, 0))
    time.sleep(.5)
    baseTee = rob.get_pose()
    #baseTee.orient =  np.array([[0,  1, 0], [ 1,  0,  0], [ 0, 0, -1]])

    eeTcam = np.array([[0, -1, 0, 0.16474797], #0.17 0.186+0.0015-0.05362+0.02715
                       [1, 0, 0, 0.001234], #0.017 -0.009-0.00375+0.013584-0.0043
                       [0, 0, 1, 0.086], #0.076 0.086
                       [0, 0, 0, 1]])

    baseTcam = np.matmul(baseTee.get_matrix(), eeTcam)
    print('pixel_row,pixel_col:', pixel_row,pixel_col)
    print('test pos', cx[pixel_row,pixel_col],cy[pixel_row,pixel_col],cz[pixel_row,pixel_col])
    scale = 1
    dg_pos_in_base = np.matmul(baseTcam, np.array([cx[pixel_row,pixel_col]*scale,cy[pixel_row,pixel_col]*scale,cz[pixel_row,pixel_col],1]))
    print('dg_pos_in_base', dg_pos_in_base)


    dg_pos_in_base[0,2] = dg_pos_in_base[0,2]+0.005

    if aperture == 2:
        gp_angle = 180 
    elif aperture == 2.5:
        gp_angle = 177 #175
    elif aperture == 3:
        gp_angle = 168
    elif aperture == 3.5:
        gp_angle = 160


    gp_control(gp_angle,100,1)
    rob.set_tcp(((aperture/100+0.0145+0.006)/2-0.006/2, 0.00, 0.3635, 0, 0, 0)) # y 0.003
    time.sleep(.5)

    #extf_angle = fg_diff_to_extf_angle(fg_dl) #+5
    if fl == 0:
        extf_angle = 180
    elif fl == 1:
        extf_angle = 140
    elif fl == 2:
        extf_angle = 100
    elif fl == 3:
        extf_angle = 40
    fg_length_control(str(int(extf_angle)))
    time.sleep(.5)

    move_acc = .5
    move_vel = .5
    rot_acc = .5
    rot_vel = .5

    eefPose = rob.get_pose()
    eefPose = eefPose.get_pose_vector()
    rob.movel((0,0,0.05+0.05,0,0,0), acc=move_acc, vel=move_vel,relative=True)
    rob.movel((dg_pos_in_base[0,0]-eefPose[0],dg_pos_in_base[0,1]-eefPose[1]-0.007,0,0,0,0), acc=move_acc, vel=move_vel,relative=True)

    #move = m3d.Transform((0,0,0,0,0,math.radians(-yaw)))
    #rob.add_pose_tool(move, acc=0.7, vel=1.2, wait=True, command="movel", threshold=None)
    #move = m3d.Transform((0,0,0,math.radians(roll),0,0))
    #rob.add_pose_tool(move, acc=0.7, vel=1.2, wait=True, command="movel", threshold=None)
    #move = m3d.Transform((0,0,0,0,math.radians(pitch),0))
    #rob.add_pose_tool(move, acc=0.7, vel=1.2, wait=True, command="movel", threshold=None)

    #self.rob.movel((0,0,pos[2]-eefPose[2],rot_z,0,0), acc=0.05, vel=0.05,relative=True)
#    move = m3d.Transform((0,0,-(dg_pos_in_base[0,2]-eefPose[2])+0.03,0,0,0))
#    rob.add_pose_tool(move, acc=0.5, vel=0.5, wait=True, command="movel", threshold=None)
    #move = m3d.Transform((0,0,0.03,0,0,0))
    #rob.add_pose_tool(move, acc=0.5, vel=0.5, wait=True, command="movel", threshold=None) # a=1, v=2
    #input("check rotate")
    # 0902 add offset
    move = m3d.Transform((0,0,0,0,0,math.radians(yaw)))


    rob.add_pose_tool(move, acc=rot_acc, vel=rot_vel, wait=True, command="movel", threshold=None)
    move = m3d.Transform((0,0,0,math.radians(roll),0,0))
    rob.add_pose_tool(move, acc=rot_acc, vel=rot_vel, wait=True, command="movel", threshold=None)
    move = m3d.Transform((0,0,0,0,math.radians(pitch),0))
    rob.add_pose_tool(move, acc=rot_acc, vel=rot_vel, wait=True, command="movel", threshold=None)
    #self.rob.movel_tool((pos[0]-eefPose[0],pos[1]-eefPose[1],pos[2]-eefPose[2],0,0,0),acc=0.05, vel=0.05)
    #rob.movel((0,0,(dg_pos_in_base[0,2]-eefPose[2])-0.035-0.05,0,0,0), acc=0.1, vel=0.1,relative=True)
    #rob.movel((0,0,(dg_pos_in_base[0,2]-eefPose[2])-0.035-0.055,0,0,0), acc=move_acc, vel=move_vel,relative=True)
    #candy: rob.movel((0,0,(dg_pos_in_base[0,2]-eefPose[2])-0.035-0.05,0,0,0), acc=0.1, vel=0.1,relative=True)
    rob.movel((0,0,(dg_pos_in_base[0,2]-eefPose[2])-0.035-0.055-0.005,0,0,0), acc=0.1, vel=0.1,relative=True)
    input('grasp?')
    #goto_rhome([106.44,-92.05,-89.54,-87.95,89.62,-72.44])
    #domino: rob.movel((0,0,(dg_pos_in_base[0,2]-eefPose[2])-0.035-0.05,0,0,0), acc=0.1, vel=0.1,relative=True)
    #go stone: rob.movel((0,0,(dg_pos_in_base[0,2]-eefPose
    #input('start digging')
    # Dig into clutter
    dig_dist = 0.035 #0.04
    dig_duration = 2.5
    safe_fz_threshold = 60
    init_fz = getFT300SensorData()[2]
    delta_z_last = 0
    delta_z_last_2 = 0
    delta_z_window = 0
    time0 = time.time()
    rob.translate_tool((0, 0, dig_dist), acc=0.03, vel=0.07, wait=False) #False

    while True:
        current_fz = getFT300SensorData()[2]
        #print(current_fz)
        delta_z = abs(current_fz - init_fz)
        delta_z_avg = (delta_z + delta_z_last + delta_z_last_2)/3
        if delta_z_window > 3:
            if delta_z_avg > safe_fz_threshold:
                print("robot test 1")
                rob.stopl()
                time.sleep(.7)
                gp_control(218,40,1) #211
                time.sleep(.3)
                grasp_success = 0
                break
            elif time.time()-time0 > dig_duration:
                time.sleep(.7)
                gp_control(218,40,1) #211
                time.sleep(.3)
                break
        delta_z_last_2 = delta_z_last
        delta_z_last = delta_z
        delta_z_window += 1

    #self.gp_control(209) # close gripper

    #self.gp_control(0.008)
    #time.sleep(.5)
    #self.go_to_home()
    #self.gp_control(0.04)
    #grasp_success = input("Is grasp successful? 1/0: ")
    rob.movel((0,0,-dg_pos_in_base[0,2]+eefPose[2]+0.15,0,0,0), acc=move_acc, vel=move_acc,relative=True)
#    grasp_success = input("Is grasp successful? 1/0: ")
    goto_rhome([106.44,-92.05,-89.54,-87.95,89.62,-72.44])
    gp_control(169,100,50)
    #input('object drop?')
    time.sleep(.1)
    goto_rhome([123.55,-81.70,-98.23,-89.59,89.67,-55.32])
    goto_c_home()
    #gp_control(169,100,50)
    time.sleep(.1)
    fg_length_control('0')
    time.sleep(.1)

    round = round + 1
    if round == 11:
        input('10')
