import numpy as np
import glob

loop_id = 0
img_save_dir = './test_data'+str(loop_id)+'/train/input/'
label_save_dir = img_save_dir.replace("input", "label")

label_list = glob.glob(label_save_dir+'*.npy')

success = 0
fail = 0
for i in range(len(label_list)):
    print(i)
    label = np.load(label_list[i])
    if label == 128:
        success+=1
    elif label ==0:
        fail+=1
success_rate = success/(success+fail)
print('success:', success_rate)