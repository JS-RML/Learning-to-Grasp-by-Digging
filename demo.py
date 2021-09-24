# -*- coding: utf-8 -*-
import numpy as np
from models import Dignet
import torch
import os
import random
import pybullet as p
import sim_class
from PIL import Image
import tool
import test_network_sim
from tqdm import tqdm

np.random.seed(0)
random.seed(0)
img_save_dir = './test_data/train/input/'
state_save_dir = img_save_dir.replace("input", "state")
random_para_save_dir = img_save_dir.replace("input", "random_para")
seg_map_dir = img_save_dir.replace("input", "seg_save")
sec_input_dir = img_save_dir.replace("input", "sec_input")

tool.create_dir_not_exist(img_save_dir)
tool.create_dir_not_exist(state_save_dir)
tool.create_dir_not_exist(random_para_save_dir)
tool.create_dir_not_exist(seg_map_dir)
tool.create_dir_not_exist(sec_input_dir)

#%%
image_pixel_before = 320
image_pixel_after = 240

def first_genrate_depth_image(floder_id):
    np.random.seed(floder_id)
    lateralFriction_random = 0.3
    globalScaling_random = 1
    
    object_type = random.randint(0,1)
    object_type = 0
    if object_type == 0:
        object_path = './objurdf/duomi/duomi.urdf'
        mass_random = 0.02
        num_obj = 130
    elif object_type == 1:
        object_path = './objurdf/gosize/cy.urdf'
        num_obj = 140
        mass_random = 0.02
    elif object_type == 2:
        object_path = './objurdf/key/sj.urdf'
        num_obj = 90+random.randint(-10,10)

    #%%
    GUI = True
    yaw_times = 6
    aps = 4
    pitch_times = 3
    roll_times = 3
    fl_times = 4


    EyePosition=[0,0,0.46]
#    EyePosition=[0,0,0.46]
    TargetPosition=[0,0,0]
    fov_d = 69.25
    near = 0.001
    far = EyePosition[2]+0.05
    state_save_path = state_save_dir+str(floder_id)+'.bullet'
    robotStartOrn = p.getQuaternionFromEuler([0, 0, 0])
    #%%
    random_para=[]
    random_para.append(GUI)
    random_para.append(num_obj)
    random_para.append(yaw_times)
    random_para.append(EyePosition)
    random_para.append(TargetPosition)
    random_para.append(fov_d)
    random_para.append(near)
    random_para.append(far)
    random_para.append(state_save_path)
    random_para.append(object_path)
    random_para.append(aps)
    random_para.append(pitch_times)
    random_para.append(roll_times)
    random_para.append(fl_times)
    #%%
    rot_step_size = 360 / yaw_times
    y_ws = np.array([rot_step_size * i for i in range(yaw_times)]).tolist()
    # 0: 0.02 1:0.03 2:0.04
    ap_ws = [0, 1, 2, 3]
    p_ws = [0,10,20]
    r_ws = [0,-10,10]
    fl_ws = [0,1,2,3]

    selected_yaw = random.sample(y_ws, 3)
    selected_pitch = random.sample(p_ws, 1)
    selected_roll= random.sample(r_ws, 1)
    selected_ap= random.sample(ap_ws, 2)
    selected_fl = random.sample(fl_ws, 2)

    random_para.append(selected_yaw)
    random_para.append(selected_pitch)
    random_para.append(selected_roll)
    random_para.append(selected_ap)
    random_para.append(selected_fl)
    
    random_para.append(mass_random)
    random_para.append(lateralFriction_random)
    random_para.append(globalScaling_random)
    
    random_para.append(object_type)
    random_para = np.array(random_para,dtype=object)
    np.save(random_para_save_dir+str(floder_id)+'.npy',random_para)
    #%%
    #_init_ sim_env
    sim = sim_class.Sim(state_save_path, num_obj, GUI, image_pixel_before, 
                        EyePosition,TargetPosition,fov_d,far,near,
                        robotStartOrn,object_path,mass_random,
                        lateralFriction_random,globalScaling_random)
    
    curr_r = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4,"video_logs/fir_task_vid_" + str(floder_id) + ".mp4") 
    #build env_sim
    sim.build_e()
    #
    rgbImg, depthImg, segImg = sim.render()
    img_d, float_depth, poke_pos_map = sim.after_render()
    img_d[np.where(segImg==0)] = 255
    p.stopStateLogging(curr_r)
    p.disconnect()
    
    return img_d, segImg


#        if not os.path.exists("video_logs/"):
#            os.makedirs("video_logs")

def poke_in_sim(row, col, yt, pt, rt, ap_ind, fl_ind,floder_id):
    random_para = np.load(random_para_save_dir+str(floder_id)+'.npy',allow_pickle=True)
    GUI = random_para[0]
    num_obj = random_para[1]
#    yaw_times = random_para[2]
    EyePosition=random_para[3]
    TargetPosition=random_para[4]
    fov_d = random_para[5]
    near = random_para[6]
    far = random_para[7]
    state_save_path = random_para[8]
    object_path = random_para[9]
    
    mass_random = random_para[19]
    lateralFriction_random = random_para[20]
    globalScaling_random = random_para[21] 
    robotStartOrn = p.getQuaternionFromEuler([0, 0, 0])
    
    
    #_init_ class
    sim = sim_class.Sim(state_save_path, num_obj, GUI, image_pixel_before, 
                        EyePosition,TargetPosition,fov_d,far,near,
                        robotStartOrn,object_path,mass_random,
                        lateralFriction_random,globalScaling_random)
    sim.restore_env()
    #image render
    rgbImg, depthImg, segImg = sim.render()
    img_d, float_depth, poke_pos_map = sim.after_render()
    img_d[np.where(segImg==0)] = 255
    ######################################################################
    
    #%%
    dig_depth = 0.03  
    
    sim.reset()
    
    curr_r = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4,"video_logs/sec_task_vid_" + str(floder_id) + ".mp4") 
    "停在上方的位置"
    surface_pos_x = poke_pos_map[row, col][0]
    surface_pos_y = poke_pos_map[row, col][1]
    surface_pos_z = poke_pos_map[row, col][2]
    
    robot_start_pos = [surface_pos_x,surface_pos_y,surface_pos_z+0.01]
    "remember r_yaw is negtive"
    r_yaw = -int(yt)
    r_pitch = int(pt)
    r_roll = int(rt)
    robot_orn = tool.world_to_gripper_orn(r_pitch, r_roll, r_yaw)

    target_pos_orn = p.multiplyTransforms(robot_start_pos, robot_orn, [0,0,-dig_depth], [0,0,0,1])

    "finger_length at row i col i"
    finger_length = str(int(fl_ind))
    "set urdf with finger_length"
    robot_path = './gripper_urdf/'+str(int(ap_ind))+finger_length+'.urdf' 
    label_at_pixel = sim.reset_and_poke(robot_start_pos,target_pos_orn,robot_orn,robot_path) 

    p.stopStateLogging(curr_r)
     
    p.disconnect()
    
    return label_at_pixel
#%%
loop_id==7  
file_id = '1VJ1uCrph1Xw9_FkU8G0pB86r14VUBcRV'
model_path = './round7.ckpt'
tool.download_file_from_google_drive(file_id, model_path)   
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('test network device', device)
model = Dignet(num_input_channels=3)
state_dict = {k.replace('auto_encoder.', ''): v for k, v in torch.load(model_path,map_location=device)['state_dict'].items()}
model.load_state_dict(state_dict)
model.to(device)
model.eval()
#%%
success = 0
fail = 0
id_ind = 0
#depth_image, seg_image = first_genrate_depth_image(floder_id=id_ind)
#tmp_img_d = depth_image.copy()
#tmp_img_d = tmp_img_d.astype(np.uint8)
#tmp_img_d = Image.fromarray(tmp_img_d)
#tmp_img_d.save('record_depth.png')
#%%
depth_image = np.array(Image.open('record_depth.png'))
pixel_row,pixel_col, yaw, pitch, roll, aperture, fl = test_network_sim.test_batch(model,depth_image,seg_image=0)
label = poke_in_sim(pixel_row,pixel_col, yaw, pitch, roll, aperture, fl,id_ind)

#id_ind = 0
#depth_image = np.array(Image.open('record_depth.png'))
#pixel_row,pixel_col, yaw, pitch, roll, aperture, fl = test_network_sim.test_batch(model,depth_image,seg_image=0)
#pixel_row = 100
#pixel_col = 100
#yaw = 240
#label = poke_in_sim(pixel_row,pixel_col, yaw, pitch, roll, aperture, fl,id_ind)

