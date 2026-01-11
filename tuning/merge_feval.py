'''
merge available full evaluation results

MD, I 26
'''
import os
import csv

DS = 'pilot'
SPLT = 'test'
DIR = 'tuning/results/'
OUT = 'merged_feval'
FOUT = f'{OUT}_{DS}_{SPLT}'

with open(f'{DIR}{FOUT}.csv', 'w', newline='') as w:
    wrt = csv.DictWriter(w, ['setting', 'Box-F1', 'mAP50', 'mAP50-95', 'Mask-F1'])
    wrt.writeheader()
    for fn in os.listdir(DIR):
        print(fn)
        if not fn.startswith(OUT):
            with open(f'{DIR}{fn}', 'r') as f:
                r = csv.DictReader(f)
                found_ds = False
                for ln in r:
                    if ln['dataset'] == DS and ln['split'] == SPLT:
                        found_ds = True
                        break
                print(ln)
                if found_ds:
                    if fn.find('.pt') == -1:
                        wrt.writerow({'setting': fn, 'Box-F1': ln['Box-F1'], 'mAP50': ln['mAP50'], 'mAP50-95': ln['mAP50-95'], 'Mask-F1': ln['Mask-F1']})
                    else:
                        wrt.writerow({'setting': fn, 'Box-F1': ln['Box-F1'], 'mAP50': ln['mAP50'], 'mAP50-95': ln['mAP50-95'], 'Mask-F1': 0})

