import numpy as np
from PIL import Image
import math
from math import sin, cos, pi
#from matplotlib import pyplot as plt
from torchvision import transforms
import torch
import torch.nn.functional as F
#from models import DenseActionSpaceDQN
from torch.utils.data import Dataset, DataLoader, random_split
import tool
import cv2
import os
from dataloder import ToTensor, FCNDataset
from tqdm import tqdm

def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)


def point_rotation2(point, rotation_pole, rot_angle):
    current_disp = list(np.array(point)-np.array(rotation_pole))
    rot_matrix = np.array([[cos(rot_angle), -sin(rot_angle)],
                           [sin(rot_angle), cos(rot_angle)]])
    current_disp = np.expand_dims(current_disp, axis=1)
    temp = np.dot(rot_matrix, current_disp)
    after = [list(rotation_pole[0]+temp[0])[0], list(rotation_pole[1]+temp[1])[0]]
    return after


def test_batch(model,img_d,seg_image):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()

    transform = transforms.Compose([
        ToTensor(),
    ])

    img_d = Image.fromarray(img_d)
    num_yaw =6
    rot_step_size = 360 / num_yaw
    yaws = np.array([rot_step_size * i for i in range(num_yaw)])
    gripper_open_ws = [0, 1, 2, 3]
    p_ws = [0,10,20]
    r_ws = [0,-10,10]
    fl_ws = [0,1,2,3]

    largest_prob = 0
    largest_prob_row = 0
    largest_prob_col = 0
    largest_pitch = 0
    largest_yaw = 0
    largest_roll = 0
    largest_op = 0
    largest_fl = 0
    largest_index = 0
    for yaw_ind in yaws:
        img_d_copy = img_d.copy()
        img_d_copy = img_d_copy.rotate(angle=yaw_ind, fillcolor = (255,255,255))
        img_d_copy = np.array(img_d_copy)
#        Image.fromarray(img_d_copy).save('./tmp/'+str(yaw_ind)+'.png')
        with torch.no_grad():
            images = transform(img_d_copy)
            images = images.unsqueeze(0)
            images = images.to(device)
            bs, c, h, w = images.shape
            outputs = model(images)
            probs = F.softmax(outputs, 1).cpu().numpy()
            # Take the good probs and good probs boolean mask
            # probs shape of [bs, C, H, W] -> good_probs and good_pred_mask shape of [bs, H, W]
#                            good_prob = probs[:, 1, :, :].copy()
            good_prob = probs[:, 1, :, :, :].copy()
            good_prob = np.squeeze(good_prob)
#            good_prob = good_prob*prob_mask*center_mask

            tmp_index, tmp_row,tmp_col = np.unravel_index(good_prob.argmax(), good_prob.shape)
            if good_prob[tmp_index][tmp_row][tmp_col]>largest_prob:
                largest_prob = good_prob[tmp_index][tmp_row][tmp_col]
                largest_prob_row = int(tmp_row)
                largest_prob_col = int(tmp_col)
                largest_yaw = int(yaw_ind)
#                largest_op = int(gripper_open_ws[gripper_open_time])*0.5+2 #cm
                largest_index = int(tmp_index)
                

    index = 0
    for pt in p_ws:
        for rt in r_ws:
            for ap_ind in gripper_open_ws:
                for fl_ind in fl_ws:
                    if index == largest_index:
                        largest_pitch = pt
                        largest_roll = rt
                        largest_op = ap_ind #cm
                        largest_fl = fl_ind   
                    index+=1
    largest_prob_row_col = point_rotation2([largest_prob_row, 240-largest_prob_col],[240/2,240/2],math.radians(int(largest_yaw)))
    final_row = int(largest_prob_row_col[0])
    final_col = int(240 - largest_prob_row_col[1])
    
#    print('largest_prob',largest_prob)
#    print('row, col',largest_prob_row,largest_prob_col)
#    print('largest_fl',largest_fl,'largest_op',largest_op)
#    print("pitch, roll, yaw", largest_pitch, largest_roll, largest_yaw )
    
    return final_row,final_col,largest_yaw, largest_pitch, largest_roll, largest_op, largest_fl


#def test_batch(model,img_d,seg_image):
#    device = 'cuda' if torch.cuda.is_available() else 'cpu'
#    print('test network device', device)
#
#    #model.eval()
#    num_yaw =6
#    rot_step_size = 360 / num_yaw
#    yaws = np.array([rot_step_size * i for i in range(num_yaw)])
#    gripper_open_ws = [0, 1, 2, 3]
#    p_ws = [0,10,20]
#    r_ws = [0,-10,10]
#    fl_ws = [0,1,2,3]
#
#    largest_prob = 0
#    largest_prob_row = 0
#    largest_prob_col = 0
#    file_name = ''
#
#    sec_input_dir = "./tmp_data/train/sec_input/"
#    img_save_dir = "./tmp_data/train/input/"
#    data_dir = "./tmp_data/"
#    tool.create_dir_not_exist(sec_input_dir)
#    tool.create_dir_not_exist(img_save_dir)
#    batch_size = 2
#    for yt in yaws:
#        for pt in p_ws:
#            for rt in r_ws:
#                for ap_ind in gripper_open_ws:
#                    for fl_ind in fl_ws:
#                        grasp_paras = np.zeros(5).reshape(1,1,5)
#                        grasp_paras[0][0][0]=ap_ind
#                        grasp_paras[0][0][1]=yt
#                        grasp_paras[0][0][2]=pt
#                        grasp_paras[0][0][3]=rt
#                        grasp_paras[0][0][4]=fl_ind
#                        grasp_paras_save_path = sec_input_dir+'num_0'+'_yaw_'+str(int(yt)) \
#                                                    +'_ap_'+str(int(ap_ind))+'_pitch_'+str(int(pt)) \
#                                                    +'_roll_'+str(int(rt))+'_fl_'+str(int(fl_ind))+'.npy'
#                        np.save(grasp_paras_save_path, grasp_paras.astype(np.int))
#                        "闃叉瑕嗙洊"
#                        tmp_img_d = img_d.copy()
#                        tmp_img_d = tmp_img_d.astype(np.uint8)
#                        tmp_img_d = Image.fromarray(tmp_img_d)
#                        tmp_img_d = tmp_img_d.rotate(angle=int(yt), fillcolor = (255,255,255))
#                        tmp_img_d_save_path = img_save_dir+'num_0'+'_yaw_'+str(int(yt)) \
#                                                    +'_ap_'+str(int(ap_ind))+'_pitch_'+str(int(pt)) \
#                                                    +'_roll_'+str(int(rt))+'_fl_'+str(int(fl_ind))+'.png'
#                        tmp_img_d.save(tmp_img_d_save_path)
#
#
#    transform = transforms.Compose([
#        ToTensor(),
#    ])
#    train_and_val = FCNDataset(data_dir=data_dir + '/train', transform=transform)
#    dl = DataLoader(train_and_val, batch_size=batch_size, num_workers=0)
#    for x1, x2, x3 in tqdm(dl):
#        with torch.no_grad():
#            outputs = model(x1.to(device), x2.to(device))
#            probs = F.softmax(outputs, 1).cpu().numpy()
#            #probs = outputs.cpu().numpy()
#            # Take the good probs and good probs boolean mask
#            # probs shape of [bs, C, H, W] -> good_probs and good_pred_mask shape of [bs, H, W]
#            good_prob = probs[:, 1, :, :].copy()
#            
#            good_prob = np.squeeze(good_prob)
#            #prob_mask = np.ones(good_prob.shape)
#            #prob_mask[:,:,np.where(seg_image<2)]=0
#            #good_prob = good_prob*prob_mask
#
#            tmp_dd, tmp_row,tmp_col = np.unravel_index(good_prob.argmax(), good_prob.shape)
#            if good_prob[tmp_dd][tmp_row][tmp_col]>largest_prob:
#                file_name = x3[tmp_dd]
#                largest_prob = good_prob[tmp_dd][tmp_row][tmp_col]
#                largest_prob_row = int(tmp_row)
#                largest_prob_col = int(tmp_col)
#    print(file_name.split('_'))
#    largest_yaw = int(file_name.split('_')[4])
#    largest_op = int(file_name.split('_')[6])
#    largest_pitch = int(file_name.split('_')[8])
#    largest_roll = int(file_name.split('_')[10])
#    largest_fl = int((file_name.split('_')[12]).split('.')[0])
#    print('largest_prob',largest_prob)
#    print('row, col',largest_prob_row,largest_prob_col)
#    print('largest_fl',largest_fl,'largest_op',largest_op)
#    print("pitch, roll, yaw", largest_pitch, largest_roll, largest_yaw )
#
#
#    largest_prob_row_col = point_rotation2([largest_prob_row, 240-largest_prob_col],[240/2,240/2],math.radians(int(largest_yaw)))
#    final_row = int(largest_prob_row_col[0])
#    final_col = int(240 - largest_prob_row_col[1])
##    tmp_pos = np.zeros(240*240).reshape(240,240)
##    tmp_pos[largest_prob_row][largest_prob_col] =1
##    tmp_pos = Image.fromarray(tmp_pos.astype(np.uint8))
##    tmp_pos = np.array(tmp_pos.rotate(angle=-int(largest_yaw)))
##    final_row,final_col = np.where(tmp_pos==1)
#    print(final_row,final_col)
#    del_file(sec_input_dir)
#    del_file(img_save_dir)
#    
#    return final_row,final_col,largest_yaw, largest_pitch, largest_roll, largest_op, largest_fl
