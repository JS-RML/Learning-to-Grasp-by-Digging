import numpy as np
from PIL import Image
import math
from math import sin, cos, pi
from matplotlib import pyplot as plt
from torchvision import transforms
import torch
import torch.nn.functional as F
from models import DenseActionSpaceDQN
from torch.utils.data import Dataset, DataLoader, random_split
#import no_erosion_aff_env as aff_env
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

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
        self.transform = transforms.ToTensor()

    def __call__(self, sample):
        image = sample
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        return self.transform(image)

def point_rotation2(point, rotation_pole, rot_angle):
    current_disp = list(np.array(point)-np.array(rotation_pole))
    rot_matrix = np.array([[cos(rot_angle), -sin(rot_angle)],
                           [sin(rot_angle), cos(rot_angle)]])
    current_disp = np.expand_dims(current_disp, axis=1)
    temp = np.dot(rot_matrix, current_disp)
    after = [list(rotation_pole[0]+temp[0])[0], list(rotation_pole[1]+temp[1])[0]]
    return after


def test4(model,img_d,prob_mask,center_mask=1):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()

    transform = transforms.Compose([
        ToTensor(),
    ])

#    bw_mask_for_prob = np.load('mask.npy')

    img_d = Image.fromarray(img_d)
    num_yaw =6
#    num_roll = 3
#    num_pitch = 3
#    gripper_open_times = 4
#    fl_times = 4
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
    for yaw_ind in tqdm(yaws):
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
            out_prob = good_prob
            good_prob = good_prob*prob_mask#*center_mask
#
#
#            "draw"
            out = good_prob
#            out = np.clip(out * 255, 0, 255).astype('uint8')
#            out = cv2.applyColorMap(out, cv2.COLORMAP_JET)
#            out_path = os.path.join( './tmp/pred_%d_%d.png' % (yaw_ind,gripper_open_time))
#            cv2.imwrite(out_path, out)

            tmp_index, tmp_row,tmp_col = np.unravel_index(good_prob.argmax(), good_prob.shape)
            if good_prob[tmp_index][tmp_row][tmp_col]>largest_prob:
                largest_prob = good_prob[tmp_index][tmp_row][tmp_col]
                largest_prob_row = int(tmp_row)
                largest_prob_col = int(tmp_col)
                largest_yaw = int(yaw_ind)
#                largest_op = int(gripper_open_ws[gripper_open_time])*0.5+2 #cm
                largest_index = int(tmp_index)
                out_for_plot = out

    index = 0
    for pt in p_ws:
        for rt in r_ws:
            for ap_ind in gripper_open_ws:
                for fl_ind in fl_ws:
                    if index == largest_index:
                        largest_pitch = pt
                        largest_roll = rt
                        largest_op = ap_ind*0.5+2 #cm
                        largest_fl = fl_ind
                    index+=1
    largest_prob_row_col = point_rotation2([largest_prob_row, 240-largest_prob_col],[240/2,240/2],math.radians(int(largest_yaw)))
    final_row = int(largest_prob_row_col[0])
    final_col = int(240 - largest_prob_row_col[1])

    print('largest_prob',largest_prob)
    print('row, col',largest_prob_row,largest_prob_col)
    print('largest_fl',largest_fl,'largest_op',largest_op)
    print("pitch, roll, yaw", largest_pitch, largest_roll, largest_yaw )

    return out_for_plot, final_row,final_col,largest_yaw, largest_pitch, largest_roll, largest_op, largest_fl, largest_index, largest_prob_row, largest_prob_col, out_prob
