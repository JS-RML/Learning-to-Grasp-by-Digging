import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import torch.nn.functional as F
from models import Dignet
import glob
import os
import random
from collections import Counter
from sys import argv
import cv2
from tqdm import tqdm
import tool

loop_id = argv[1]
comp_id = argv[2]
example_number_need_collect = int(argv[3])

#loop_id = 3
#comp_id = 'l'
#example_number_need_collect = 1

img_save_dir = './data'+str(loop_id)+'/train/input/'
label_save_dir = img_save_dir.replace("input", "label")
state_save_dir = img_save_dir.replace("input", "state")
random_para_save_dir = img_save_dir.replace("input", "random_para")
tqdm_dir = img_save_dir.replace("input", "tqdm_p")
predict_save_dir = img_save_dir.replace("input", "predict_save")
seg_map_dir = img_save_dir.replace("input", "seg_save")
sec_input_dir = img_save_dir.replace("input", "sec_input")


#%%
use_pretrain = False
all_random_cho = 50
random_choice = 10
nn_choice = 40

#%%
image_pixel_after = 240
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#%%
if use_pretrain == True and int(loop_id)==0:
    #"start downloading pretrain .ckpt"
    model_path = './pretrain.ckpt'
#elif int(loop_id)==4:    
#    model_path = './round3.ckpt'   
#elif int(loop_id)==5:    
#    model_path = './round4.ckpt'   
elif int(loop_id)==7:    
    file_id = '1VJ1uCrph1Xw9_FkU8G0pB86r14VUBcRV'
    model_path = './round7.ckpt'
    tool.download_file_from_google_drive(file_id, model_path)     
#%%   
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
        self.transform = transforms.ToTensor()

    def __call__(self, sample):
        image= sample
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        return self.transform(image)

def load_model():
    model = Dignet(num_input_channels=3)
    state_dict = {k.replace('auto_encoder.', ''): v for k, v in torch.load(model_path,map_location='cpu')['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def get_img_path_list():
    img_path_list = glob.glob(img_save_dir+'*.png')
    return img_path_list


def test(model):
    transform = transforms.Compose([
        ToTensor(),
    ])

    img_paths = get_img_path_list()
    #%%
    num_yaw =6
    rot_step_size = 360 / num_yaw
    yaws = np.array([rot_step_size * i for i in range(num_yaw)])
    ap_ws = [0, 1, 2, 3]
    p_ws = [0,10,20]
    r_ws = [0,-10,10]
    fl_ws = [0,1,2,3]
    #%%
    for img_path in tqdm(img_paths):
        fname =os.path.split(img_path)[1].split('.')[0]
#        print(fname)
#        print(fname.split('_'))
        target_yaw = fname.split('_')[3]
        target_ap = fname.split('_')[5]
        target_pitch = fname.split('_')[7]
        target_roll = fname.split('_')[9]
        target_fl = fname.split('_')[11]
#        print(target_yaw,target_ap,target_pitch,target_roll,target_fl)
        #%%
        tmp_index = 0
        index = 0
        for yaw_ind in yaws: 
            for pt in p_ws:
                for rt in r_ws:
                    for ap_ind in ap_ws:
                        for fl_ind in fl_ws:
                            if yaw_ind==target_yaw and pt==target_pitch and rt == target_roll and ap_ind == target_ap and fl_ind == target_fl:
                                tmp_index = index
                                break
                            index+=1
                        #%%
        origin_angle = fname.split('_')[3].split('.')[0]
        img_d= Image.open(img_path)
        with torch.no_grad():
            images = transform(img_d)
            images = images.unsqueeze(0)
            images = images.to(device)
            bs, c, h, w = images.shape
            outputs = model(images)
            probs = F.softmax(outputs, 1).cpu().numpy()
            # Take the good probs and good probs boolean mask
            # probs shape of [bs, C, H, W] -> good_probs and good_pred_mask shape of [bs, H, W]
            good_prob = probs[:, 1, tmp_index, :, :].copy()
            
            good_prob = np.squeeze(good_prob)
            predict_index = np.unravel_index(np.argpartition(good_prob.ravel(), -nn_choice)[-nn_choice:], good_prob.shape)
            predict_poke_pos = np.zeros(image_pixel_after*image_pixel_after).reshape(image_pixel_after,image_pixel_after)
            predict_poke_pos[predict_index]=1



            all_pokeable_pos = Image.open((img_path[:-4] + '.png').replace('input', 'seg_save'))
            all_pokeable_pos = np.array(all_pokeable_pos)
            all_pokeable_pos[predict_index]=0

            #delete bowl and plane
            all_pokeable_pos[np.where(all_pokeable_pos<2)] = 0
            #count pixel number
            counter_dict = Counter(all_pokeable_pos[np.where(all_pokeable_pos>0)])
            counter_dict = {k : v for k, v in counter_dict.items() if v < 250}
            for key in list(counter_dict.keys()):
                all_pokeable_pos[np.where(all_pokeable_pos==key)] = 0
            #erod
            kernel = np.ones((3,3), np.uint8)
            all_pokeable_pos = cv2.erode(all_pokeable_pos, kernel)

            exclude_poke_pos = np.where(all_pokeable_pos!=0)
            # SHUFFULE AND SELECT
            lis = range(len(exclude_poke_pos[0]))
            lis = list(lis)
            random.shuffle(lis)
            lis= lis[0:random_choice]
            for eppi in lis:
                predict_poke_pos[exclude_poke_pos[0][eppi]][exclude_poke_pos[1][eppi]]=1

            "label, -origin_angle"
            predict_poke_pos = Image.fromarray(predict_poke_pos)
            predict_poke_pos = predict_poke_pos.rotate(angle=-int(origin_angle), fillcolor = (0))
            predict_poke_pos = np.array(predict_poke_pos)
            np.save(predict_save_dir+fname+'.npy', predict_poke_pos.astype(np.int))
            
            #####
#            predict_poke_pos[np.where(predict_poke_pos==1)]=255
#            predict_poke_pos = Image.fromarray(predict_poke_pos.astype(np.uint8))
#            predict_poke_pos = predict_poke_pos.rotate(angle=int(origin_angle), fillcolor = (0))
#            predict_poke_pos.save(predict_save_dir+fname+'.png',mode='L')
#            
##            ####
#            out = good_prob
#            print(np.max(out))
#            out = cv2.applyColorMap((out*255).astype(np.uint8), cv2.COLORMAP_JET)
#            out_path = os.path.join(predict_save_dir+fname+'p.png')
#            cv2.imwrite(out_path, out)
#            imgd_copy = np.array(img_d).copy()
#            prediction_vis = (0.6*imgd_copy + 0.4*out).astype(np.uint8)
#            out_path = os.path.join(predict_save_dir+fname+'p2.png')
#            cv2.imwrite(out_path, prediction_vis)
#

def all_random():

    img_paths = get_img_path_list()
    for img_path in tqdm(img_paths):
        fname =os.path.split(img_path)[1].split('.')[0]
        yaw_angle = fname.split('_')[3].split('.')[0]
        predict_poke_pos = np.zeros(image_pixel_after*image_pixel_after).reshape(image_pixel_after,image_pixel_after)

        seg_map_path =seg_map_dir+fname+'.png'
        #load seg_map
        all_pokeable_pos = Image.open(seg_map_path)
        all_pokeable_pos = np.array(all_pokeable_pos)
        #delete bowl and plane
        all_pokeable_pos[np.where(all_pokeable_pos<2)] = 0
        #count pixel number
        counter_dict = Counter(all_pokeable_pos[np.where(all_pokeable_pos>0)])
        counter_dict = {k : v for k, v in counter_dict.items() if v < 250}
        for key in list(counter_dict.keys()):
            all_pokeable_pos[np.where(all_pokeable_pos==key)] = 0
        exclude_poke_pos = np.where(all_pokeable_pos!=0)
        # erod
        kernel = np.ones((3,3), np.uint8)
        all_pokeable_pos = cv2.erode(all_pokeable_pos, kernel)
        # shufle and select
        lis = range(len(exclude_poke_pos[0]))
        lis = list(lis)
        random.shuffle(lis)
        lis= lis[0:all_random_cho]
        for eppi in lis:
            predict_poke_pos[exclude_poke_pos[0][eppi]][exclude_poke_pos[1][eppi]]=1
        "label, -origin_angle"
        predict_poke_pos = Image.fromarray(predict_poke_pos)
        predict_poke_pos = predict_poke_pos.rotate(angle=-int(yaw_angle), fillcolor = (0))
        predict_poke_pos = np.array(predict_poke_pos)
        np.save(predict_save_dir+fname+'.npy', predict_poke_pos.astype(np.int32))

#model=load_model()
#test(model)
if __name__ == '__main__':
#    print(time.localtime(time.time()))
    if use_pretrain == False and int(loop_id)==0:
        all_random()
        print('total random')
    else:
        model=load_model()
        test(model)

