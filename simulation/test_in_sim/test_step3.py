# import os
#import time
import tool
import pybullet as p
import numpy as np
import sim_class
# from PIL import Image
#import random
#import multiprocessing
# from sys import argv
import concurrent.futures
# import gc 
import random

loop_id = 0
comp_id = 0
example_number_need_collect = 5000

img_save_dir = './test_data'+str(loop_id)+'/train/input/'
label_save_dir = img_save_dir.replace("input", "label")
state_save_dir = img_save_dir.replace("input", "state")
random_para_save_dir = img_save_dir.replace("input", "random_para")
tqdm_dir = img_save_dir.replace("input", "tqdm_p")
predict_save_dir = img_save_dir.replace("input", "predict_save")
seg_map_dir = img_save_dir.replace("input", "seg_save")
sec_input_dir = img_save_dir.replace("input", "sec_input")

#%%

#%%
image_pixel_before = 320
image_pixel_after = 240

def custom_method(floder_id):
    # np.random.seed(floder_id)
    # random.seed(floder_id)
    predict_save = np.load(predict_save_dir+str(int(floder_id))+'.npy')
    index = predict_save[0]
    row = int(predict_save[1])
    col = int(predict_save[2])
    yt = int(predict_save[3])
    pt = int(predict_save[4])
    rt = int(predict_save[5])
    ap_ind = int(predict_save[6])
    fl_ind = int(predict_save[7])
    
    random_para = np.load(random_para_save_dir+str(floder_id)+'.npy',allow_pickle=True)
    GUI = random_para[0]
    num_obj = int(random_para[1])
    yaw_times = random_para[2]
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
    dig_depth = 0.04   
    
    sim.reset()
       
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
    p.disconnect()
    
    label_save_path =label_save_dir+ str(index)+'.npy'
    np.save(label_save_path, label_at_pixel)
            
#for i in range(33,100):
    #print(i)
    #custom_method(33)  
    
if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(custom_method,floder_id) for floder_id in range(example_number_need_collect)]
        for future in concurrent.futures.as_completed(futures):
              try:
                  print(future.result())
              except Exception as exc:
                  print(f'Generated an exception: {exc}')
    
