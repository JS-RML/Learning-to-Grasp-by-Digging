import subprocess
import time
import gc


loop_id = 0

print(loop_id,'domino start')
print (time.asctime( time.localtime(time.time()) ))
subprocess.check_output(['python', 'pretrain_domino.py', str(loop_id)])
gc.collect()


print(loop_id,'key start')
print (time.asctime( time.localtime(time.time()) ))
subprocess.check_output(['python', 'pretrain_sj.py', str(loop_id)])
gc.collect()

print(loop_id,'gostone start')
print (time.asctime( time.localtime(time.time()) ))
subprocess.check_output(['python', 'pretrain_cy.py', str(loop_id)])
gc.collect()

'''
label_namelist = glob.glob('./pre_data'+str(loop_id)+'/train/label/*.png')
input_namelist = glob.glob('./pre_data'+str(loop_id)+'/train/label/*.png')
sec_namelist = glob.glob('./pre_data'+str(loop_id)+'/train/label/*.npy')
for i in range(len(label_namelist)):
    label = np.array(Image.open(label_namelist[i]))
    if np.min(label) == 255:
        os.remove(label_namelist[i])
        os.remove(input_namelist[i])
        os.remove(sec_namelist[i])
'''
