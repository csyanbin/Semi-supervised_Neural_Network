import numpy as np
import os

dataset = 'hela'
ratio = 1
if not ratio==1.0:
    log_dir = dataset+'/log_'+str(ratio)+'/'
else:
    log_dir = dataset+'/log_1/'

acc_array= np.zeros((6,20))
for label in range(1,7):
    for rand in range(1,21):
        if not ratio==1.0:
            log_name = log_dir + 'log_label'+str(label)+'_'+str(rand)+'_'+str(ratio)+'.log'
        else:
            log_name = log_dir + 'log_label'+str(label)+'_'+str(rand)+'_'+str(ratio)+'.log'
        
        if not os.path.exists(log_name):
            break
        lines = open(log_name).readlines()
        for line in lines:
            if line.startswith('Epoch: 50'):
                line = line.strip()
                ind = line.strip().rfind('Best test:')+len('Best test:')
                acc = float(line[ind:])
                print(acc)
                acc_array[label-1][rand-1] = acc
                break

print(acc_array)
print np.mean(acc_array, 1)
print np.std(acc_array, 1)
