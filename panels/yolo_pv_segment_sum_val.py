'''
Docstring for panels.yolo_pv_segment_sum_val

YOLO segmentation inference with results transformation
in a GPU-efficient scheme
provides visual results for a selected subset of images

MD

I 26
'''

from torch import squeeze
from time import time
from ultralytics import YOLO
from os import listdir


PROJ = 'PV_inf_check'
data = 'imagery125_640px/'
out = 'pv_areas_125_mick.csv'
mod = 'ft26_SGD_finloop.pt'
v = True
batch_size = 256


def calculate_area_share_batch(imgs, model: YOLO, out_f: str):
    t = time()
    yolo_results = model.predict(imgs, device='cuda:0', verbose=v, batch=batch_size, stream=True, save=True, project=PROJ, show_labels=False)
    print('setting up batch-streamed YOLO inference took', time()-t)
    t = time()
    for res in yolo_results: # 1 per img, so n in total
        if res is not None and res.masks is not None: # sth was detected
            pv_area = 0
            img_w, img_h = 640, 640 # res.orig_shape
            ppth = res.path
            mask_sum = squeeze(res.masks.data).sum() # [1, 640, 640]
            pv_area = mask_sum.div(img_w*img_h) # percentage
            print(f'{time()-t},{ppth},{pv_area}\n')
            with open(out_f, 'a') as f:
                f.write(f'{ppth},{pv_area}\n')
    print('saving batch results took', time()-t)


if __name__ == '__main__':
    model = YOLO(mod)
    all_files = [f'{data}/{f}' for f in listdir(data) if f.find('N-34-50-C-c-4-4') != -1]
    calculate_area_share_batch(all_files, model, out)
    