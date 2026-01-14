'''
prepend appropriate header to feval csv files

MD, I 26
'''
import os

DIR = 'tuning/results/'
                    
for fn in os.listdir(DIR):
    with open(f'{DIR}{fn}', 'r+') as f:
        c = f.read()
        if fn.find('merge') == -1 and c.find('dataset') == -1:
            print(fn)
            f.seek(0, 0)
            if fn.find('.ariel') == -1:
                f.write('dataset,split,Class,Images,Instances,Box-P,Box-R,Box-F1,mAP50,mAP50-95,Mask-P,Mask-R,Mask-F1,t\n')
            else:
                f.write('dataset,split,Class,Images,Instances,Box-P,Box-R,Box-F1,mAP50,mAP50-95,t\n')
            f.write(c)
    
