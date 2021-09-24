import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import torch.nn.functional as F
from models import DenseActionSpaceDQN
import glob
import math
from torch.utils.data import Dataset, DataLoader
from math import sin, cos
from tqdm import tqdm
import cv2
import os
import tool
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
image_pixel_after = 240
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#%%
file_id = '1ss3IqsKzYgxnZKIlSxua8zBRSoSnDZSC'
model_path = './round3.ckpt'
tool.download_file_from_google_drive(file_id, model_path)   
#%%
def point_rotation2(point, rotation_pole, rot_angle):
    current_disp = list(np.array(point)-np.array(rotation_pole))
    rot_matrix = np.array([[cos(rot_angle), -sin(rot_angle)],
                           [sin(rot_angle), cos(rot_angle)]])
    current_disp = np.expand_dims(current_disp, axis=1)
    temp = np.dot(rot_matrix, current_disp)
    after = [list(rotation_pole[0]+temp[0])[0], list(rotation_pole[1]+temp[1])[0]]
    return after
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

class FCNDataset(Dataset):
    def __init__(self, data_dir='./fcn_sampledata', transform=None):
        """Custom dataset class
        
        Args
        :data_dir: Path to data directory
        :transform: Transform 
        
        Attributes
        :data_dir:
        :transform:
        :image_filenames:
        :label_filenames:
        """
        self.data_dir = data_dir
        self.transform = transform
        
        self.input1 = glob.glob(data_dir)
        # self.input2 = glob.glob(os.path.join(data_dir, 'sec_input', '*'))
        # self.input2 = [path.replace('input/', 'sec_input/').replace('.png', '.npy') for path in self.input1]
        # self.label_filenames = glob.glob(os.path.join(data_dir, 'label', '*'))
        self.input2 = []
        for path in self.input1:
            self.input2.append((path[:-4] + '.npy').replace('input', 'sec_input'))
    
    
    def __len__(self):
        return len(self.input1)
    
    def __getitem__(self, idx):
        input1 = self.input1[idx]
        input2_file = self.input2[idx]
        
        image = np.array(Image.open(input1))
        input2 = np.load(input2_file).astype('float32')
        if self.transform:
            image = self.transform(image)
        
        input2 = torch.from_numpy(input2)
        
#        input3_fname = torch.from_numpy(np.array(input1))
        # print(image.dtype, input2.dtype, label.dtype)
        return image, input2, input1



#%%
def load_model():
    model = DenseActionSpaceDQN(num_input_channels=3)
    state_dict = {k.replace('auto_encoder.', ''): v for k, v in torch.load(model_path,map_location='cpu')['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def get_img_path_list():
    img_path_list = glob.glob(img_save_dir+'*.png')
    return img_path_list


def test_batch(model,img_d):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform = transforms.Compose([
        ToTensor(),
    ])

    img_d = Image.fromarray(img_d)
    
    img_d = img_d.resize((240, 240),Image.ANTIALIAS)
    
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
                
                
#            out = good_prob[largest_index,:,:]
#            out = np.squeeze(out)
#            print(np.max(out))
#            out = cv2.applyColorMap((out*255).astype(np.uint8), cv2.COLORMAP_JET)
#            out_path = str(yaw_ind)+'p.png'
#            cv2.imwrite(out_path, out)
#            imgd_copy = np.array(img_d).copy()
#            prediction_vis = (0.6*imgd_copy + 0.4*out).astype(np.uint8)
#            out_path = str(yaw_ind)+'p2.png'
#            cv2.imwrite(out_path, prediction_vis)
            
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
    
    return final_row,final_col,largest_yaw, largest_pitch, largest_roll, largest_op, largest_fl


    # print(index, final_row,final_col,largest_yaw, largest_pitch, largest_roll, largest_op, largest_fl)    


model=load_model()
model.eval()
for floder_id in tqdm(range(example_number_need_collect)):
    img_d = np.array(Image.open(img_save_dir+'num_'+str(floder_id)+'.png'))
    final_row,final_col,largest_yaw, largest_pitch, largest_roll, largest_op, largest_fl = test_batch(model, img_d)
    
    predict_save = []
    predict_save.append(floder_id)
    predict_save.append(final_row)
    predict_save.append(final_col)
    predict_save.append(largest_yaw)
    predict_save.append(largest_pitch)
    predict_save.append(largest_roll)
    predict_save.append(largest_op)
    predict_save.append(largest_fl)
    np.save(predict_save_dir+str(int(floder_id))+'.npy', predict_save)




