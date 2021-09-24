import subprocess
import time
import gc
   

       
print('step1 start')
print (time.asctime( time.localtime(time.time()) ))
subprocess.check_output(['python', './src/test_in_sim/test_step1.py'])



print('step2 start')
print (time.asctime( time.localtime(time.time()) ))
subprocess.check_output(['python', './src/test_in_sim/test_step2.py'])
gc.collect()

print('step3 start')
print (time.asctime( time.localtime(time.time()) ))
subprocess.check_output(['python', './src/test_in_sim/test_step3.py'])


print('step4 start')
print (time.asctime( time.localtime(time.time()) ))
subprocess.check_output(['python', './src/test_in_sim/test_step4.py'])


