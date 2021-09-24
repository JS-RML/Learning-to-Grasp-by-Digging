import subprocess
import time
import multiprocessing 


loop_id = 7
collect_n = 5000

if multiprocessing.cpu_count() == 128:
    comp_id = '3a'+str(loop_id)+'r'
elif multiprocessing.cpu_count() == 64:
    comp_id = '3b'+str(loop_id)+'r'
elif multiprocessing.cpu_count() == 32:
    comp_id = '3c'+str(loop_id)+'r'
    collect_n = int(collect_n/2)
      
else:
    comp_id = 'l'
    collect_n = 1

print(loop_id,'step1 start')
print (time.asctime( time.localtime(time.time()) ))
subprocess.check_output(['python', './src/train_in_sim/step1.py', str(loop_id), str(comp_id), str(collect_n)])

print(loop_id,'step2 start')
print (time.asctime( time.localtime(time.time()) ))
subprocess.check_output(['python', './src/train_in_sim/step2.py', str(loop_id), str(comp_id), str(collect_n)])


print(loop_id,'step3 start')
print (time.asctime( time.localtime(time.time()) ))
subprocess.check_output(['python', './src/train_in_sim/step3.py', str(loop_id), str(comp_id), str(collect_n)])


print(loop_id,'step4 start')
subprocess.check_output(['python', './src/train_in_sim/arrange_data.py', str(loop_id), str(comp_id)])
print (time.asctime( time.localtime(time.time()) ))

print(loop_id,'step5 start')
print (time.asctime( time.localtime(time.time()) ))
subprocess.check_output(['python', './src/train_in_sim/step4.py', str(loop_id), str(comp_id), str(collect_n)])
