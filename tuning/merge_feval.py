'''
Docstring for tuning.merge_feval

merge available full evaluation results

MD, I 26
'''
import os
import csv

DS = 'rzeszow'
SPLT = 'train'
LN = 9 # rzeszow-train, early ft saved differently (ft4 would be incorrect)
DIR = 'tuning/results/'
OUT = 'merged_feval.csv'

with open(f'{DIR}{OUT}', 'w', newline='') as w:
    wrt = csv.DictWriter(w, ['setting', 'Box-F1', 'mAP50', 'Mask-F1'])
    wrt.writeheader()
    for fn in os.listdir(DIR):
        print(fn)
        if fn != OUT:
            with open(f'{DIR}{fn}', 'r') as f:
                r = csv.DictReader(f)
                ln = list(r)[LN]
                wrt.writerow({'setting': fn, 'Box-F1': ln['Box-F1'], 'mAP50': ln['mAP50'], 'Mask-F1': ln['Mask-F1']})

