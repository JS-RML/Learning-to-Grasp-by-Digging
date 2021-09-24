import glob
import os
import numpy as np
import shutil
from tqdm import tqdm 
from PIL import Image
import tool
import os.path
from sys import argv

loop_id = argv[1]
comp_id = argv[2]
for comp in range(3):
    if comp ==0:
        comp_id='3a'
        data_from_dir = './3adata'+str(int(loop_id))+'/data'+str(int(loop_id))+'/train/'
    if comp ==1:
        comp_id='3b'
        data_from_dir = './3bdata'+str(int(loop_id))+'/data'+str(int(loop_id))+'/train/'
    if comp ==2:
        comp_id='3c'
        data_from_dir = './3cdata'+str(int(loop_id))+'/data'+str(int(loop_id))+'/train/'
    input_namelist = glob.glob(data_from_dir+ 'input/*.png')
    count =0  
    for path in tqdm(input_namelist):
        label_name = (path[:-4] + '.png').replace('input', 'label')
        if os.path.isfile(label_name)  == False:
            count +=1
            os.remove(path)            
for comp in range(3):
    if comp ==0:
        comp_id='3a'+str(int(loop_id))+'r'
        data_from_dir = './3adata'+str(int(loop_id))+'/data'+str(int(loop_id))+'/train/'
    if comp ==1:
        comp_id='3b'+str(int(loop_id))+'r'
        data_from_dir = './3bdata'+str(int(loop_id))+'/data'+str(int(loop_id))+'/train/'
    if comp ==2:
        comp_id='3c'+str(int(loop_id))+'r'
        data_from_dir = './3cdata'+str(int(loop_id))+'/data'+str(int(loop_id))+'/train/' 
        
    input_namelist = glob.glob(data_from_dir+ 'input/*.png')
    label_namelist=[]
    for path in input_namelist:
        label_name = (path[:-4] + '.png').replace('input', 'label')
        label_namelist.append(label_name)
        
    scene_num = int(len(input_namelist)/12)
    
    
    yaw_times = 6
    rot_step_size = 360 / yaw_times
    y_ws = np.array([rot_step_size * i for i in range(yaw_times)]).tolist()
    ap_ws = [0, 1, 2, 3]
    p_ws = [0,10,20]
    r_ws = [0,-10,10]
    fl_ws = [0,1,2,3]
    
    image_save_dir = './tmp_data/input/'
    label_save_dir = './tmp_data/label/'
    tool.create_dir_not_exist(image_save_dir)
    tool.create_dir_not_exist(label_save_dir)
    
    for i in tqdm(range(scene_num)):
        tmp_new_label = np.ones(240*240*144,dtype=int).reshape(240,240,144)*255
        tmp_need_move_filename = ''        
        for yt in y_ws: 
            index = 0
            for pt in p_ws:
                for rt in r_ws:
                    for ap_ind in ap_ws:
                        for fl_ind in fl_ws:
                            search_labelname = data_from_dir+'label/'+str(comp_id)+'num_'+str(int(i))+'_yaw_'+str(int(yt)) \
                                                        +'_ap_'+str(int(ap_ind))+'_pitch_'+str(int(pt)) \
                                                        +'_roll_'+str(int(rt))+'_fl_'+str(int(fl_ind))+'.png'
                            if os.path.exists(search_labelname) == True:
                                tmp_new_label[:,:,index] = np.asarray(Image.open(search_labelname))
                                np.save(label_save_dir+comp_id+'_'+str(int(i))+'_yaw_'+str(int(yt))+'.npy',tmp_new_label.astype(np.uint8))
    #                            print(np.sum(tmp_new_label))
                            search_imagename = data_from_dir+'input/'+str(comp_id)+'num_'+str(int(i))+'_yaw_'+str(int(yt)) \
                                +'_ap_'+str(int(ap_ind))+'_pitch_'+str(int(pt)) \
                                +'_roll_'+str(int(rt))+'_fl_'+str(int(fl_ind))+'.png'
                                
                            if os.path.exists(search_imagename) == True:
                                tmp_need_move_filename = search_imagename
                                shutil.copy(tmp_need_move_filename, image_save_dir+comp_id+'_'+str(int(i))+'_yaw_'+str(int(yt))+'.png')
                                
                            index+=1
        
        
    
    
    
    


    