import os
#import time
import tool
import pybullet as p
import numpy as np
import sim_class
from PIL import Image
#import random
#import multiprocessing
from sys import argv
import concurrent.futures
import gc 


loop_id = argv[1]
comp_id = argv[2]
example_number_need_collect = int(argv[3])

#loop_id = 0
#comp_id = 0
#example_number_need_collect = 1

img_save_dir = './data'+str(loop_id)+'/train/input/'
label_save_dir = img_save_dir.replace("input", "label")
state_save_dir = img_save_dir.replace("input", "state")
random_para_save_dir = img_save_dir.replace("input", "random_para")
tqdm_dir = img_save_dir.replace("input", "tqdm_p")
predict_save_dir = img_save_dir.replace("input", "predict_save")
seg_map_dir = img_save_dir.replace("input", "seg_save")
sec_input_dir = img_save_dir.replace("input", "sec_input")

img_save_dir = img_save_dir+str(comp_id)
label_save_dir = label_save_dir +str(comp_id)
state_save_dir = state_save_dir+str(comp_id)
random_para_save_dir = random_para_save_dir +str(comp_id)
tqdm_dir = tqdm_dir+str(comp_id)
predict_save_dir = predict_save_dir+str(comp_id)
seg_map_dir = seg_map_dir+str(comp_id)
sec_input_dir = sec_input_dir+str(comp_id)

#%%

#%%
image_pixel_before = 320
image_pixel_after = 240
 
def custom_method(floder_id):
    random_para = np.load(random_para_save_dir+str(floder_id)+'.npy',allow_pickle=True)
    GUI = random_para[0]
    num_obj = random_para[1]
    yaw_times = random_para[2]
    EyePosition=random_para[3]
    TargetPosition=random_para[4]
    fov_d = random_para[5]
    near = random_para[6]
    far = random_para[7]
    state_save_path = random_para[8]
    object_path = random_para[9]
#    aps = random_para[10]
#    pitch_times = random_para[11]
#    roll_times = random_para[12]
#    fl_times = random_para[13]    
    selected_yaw = random_para[14]
    selected_pitch = random_para[15]
    selected_roll= random_para[16]
    selected_ap= random_para[17]
    selected_fl = random_para[18]
    
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
    img_d[np.where(segImg==0)] = 0
    ######################################################################
    #poke_pos_map
       
    label_npy = np.ones(image_pixel_after*image_pixel_after,dtype=int).reshape(image_pixel_after,image_pixel_after)*255
    
    #%%
#    rot_step_size = 360 / yaw_times
#    y_ws = np.array([rot_step_size * i for i in range(yaw_times)]).tolist()
    # 0: 15cm 1: 25cm 2: 35cm 3: 45cm
#    ap_ws = [0, 1, 2, 3]
#    p_ws = [0,10,20]
#    r_ws = [0,-10,10]
    # 0: 0cm 1: 1cm 2: 2cm 3: 3cm
#    fl_ws = [0,1,2,3]
    dig_depth = 0.04  
    count_poke = 0
    

    
    for yt in selected_yaw:
        for pt in selected_pitch:
            for rt in selected_roll:
                for ap_ind in selected_ap:   
                    for fl_ind in selected_fl:
                        predict_points_map_path = predict_save_dir+'num_'+str(floder_id)+'_yaw_'+str(int(yt)) \
                                                    +'_ap_'+str(int(ap_ind))+'_pitch_'+str(int(pt)) \
                                                    +'_roll_'+str(int(rt))+'_fl_'+str(int(fl_ind))+'.npy'
                        predict_points_map = np.load(predict_points_map_path)  
                        
                        tmp_label_npy = label_npy.copy()    
                        row_list = np.where(predict_points_map==1)[0]
                        col_list = np.where(predict_points_map==1)[1]
                        
                        for row, col in zip(row_list,col_list):
                            count_poke +=1
                            sa_c_p = tqdm_dir +'_floder_id_' +str(floder_id)+'_count_'+str(count_poke)+'.npy'
                            np.save(sa_c_p,count_poke)
                            if count_poke>1:
                                os.remove(tqdm_dir +'_floder_id_' +str(floder_id)+'_count_'+str(count_poke-1)+'.npy') 
#                
                            sim.reset()
                               
                            ""
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
                            
                                
                        #    print(label_at_pixel)
                            tmp_label_npy[row][col]=int(label_at_pixel)
                        
                        tmp_label_npy = tmp_label_npy.astype(np.uint8)
                        tmp_label_npy = Image.fromarray(tmp_label_npy)
                        tmp_label_npy=tmp_label_npy.rotate(angle=int(yt), fillcolor = (255))
                        tmp_label_npy_s_path =label_save_dir+'num_'+str(floder_id)+'_yaw_'+str(int(yt)) \
                                                    +'_ap_'+str(int(ap_ind))+'_pitch_'+str(int(pt)) \
                                                    +'_roll_'+str((rt))+'_fl_'+str(int(fl_ind))+'.png'
                        tmp_label_npy.save(tmp_label_npy_s_path,mode='L')
                    
    
    
    p.disconnect()
    gc.collect()
            
            
if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(custom_method,floder_id) for floder_id in range(example_number_need_collect)]
        for future in concurrent.futures.as_completed(futures):
             try:
                 print(future.result())
             except Exception as exc:
                 print(f'Generated an exception: {exc}')
