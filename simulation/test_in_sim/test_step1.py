import random
#import time
import pybullet as p
import numpy as np
import sim_class
from PIL import Image
import concurrent.futures
import tool
# from sys import argv
import gc
#import multiprocessing


loop_id = 0
comp_id = 0
example_number_need_collect = 5000

img_save_dir = './test_data'+str(loop_id)+'/train/input/'
label_save_dir = img_save_dir.replace("input", "label")
state_save_dir = img_save_dir.replace("input", "state")
random_para_save_dir = img_save_dir.replace("input", "random_para")
tqdm_dir = img_save_dir.replace("input", "tqdm_p")
predict_save_dir = img_save_dir.replace("input", "predict_save")
#seg_map_dir = img_save_dir.replace("input", "seg_save")
#sec_input_dir = img_save_dir.replace("input", "sec_input")

tool.create_dir_not_exist(img_save_dir)
tool.create_dir_not_exist(label_save_dir)
tool.create_dir_not_exist(state_save_dir)
tool.create_dir_not_exist(random_para_save_dir)
tool.create_dir_not_exist(tqdm_dir)
tool.create_dir_not_exist(predict_save_dir)
#tool.create_dir_not_exist(seg_map_dir)
#tool.create_dir_not_exist(sec_input_dir)

#%%
image_pixel_before = 320
image_pixel_after = 240
#%%
def custom_method_saveimg(floder_id):
    np.random.seed(floder_id)
    random.seed(floder_id)
    object_type = 0
    if object_type == 0:
        object_path = './objurdf/duomi/duomi.urdf'
        num_obj = 140
        mass_random = 0.006
    elif object_type == 1:
        object_path = './objurdf/cy/cy.urdf'
        num_obj = 150
        mass_random = 0.0045
    elif object_type == 2:
        object_path = './objurdf/sj/sj.urdf'
        num_obj = 90

    #%%
    GUI = False
    yaw_times = 6
    aps = 4
    pitch_times = 3
    roll_times = 3
    fl_times = 4


    EyePosition=[0,0,0.46+random.uniform(-0.01,0.01)]
    TargetPosition=[0,0,0]
    fov_d = 69.25+random.uniform(-0.25,0.25)
    near = 0.001
    far = EyePosition[2]+0.05
    state_save_path = state_save_dir+str(int(floder_id))+'.bullet'
    robotStartOrn = p.getQuaternionFromEuler([0, 0, 0])
    
    
    lateralFriction_random = random.uniform(0.25,0.35)
    globalScaling_random = random.uniform(0.95,1)
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
    #build env_sim
    sim.build_e()
    #渲染图像
    rgbImg, depthImg, segImg = sim.render()
    img_d, float_depth, poke_pos_map = sim.after_render()
    img_d[np.where(segImg==0)] = 255

    tmp_img_d = img_d.copy()
    tmp_img_d = tmp_img_d.astype(np.uint8)
    tmp_img_d = Image.fromarray(tmp_img_d)
    tmp_img_d_save_path = img_save_dir+'num_'+str(floder_id)+'.png'
    tmp_img_d.save(tmp_img_d_save_path)


    p.disconnect()
    gc.collect()

if __name__ == '__main__':
#    print(time.localtime(time.time()))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(custom_method_saveimg,floder_id) for floder_id in range(example_number_need_collect)]
        for future in concurrent.futures.as_completed(futures):
             try:
                 pass
             except Exception as exc:
                 print(f'Generated an exception: {exc}')
