'''
Docstring for panels.yolo_pv_segment_sum

YOLO segmentation inference with results transformation
in a GPU-efficient scheme

MD

I 26
'''

import torch
from time import time
from ultralytics import YOLO


data = 'divided_images_jpg/'
data = 'dane_syntetyczne/images/'
data = 'imagery125_640px/'
out = 'pv_areas_125.csv'
mod = 'ft26_SGD_finloop.pt'
v = True
batch_size = 256


def calculate_area_share_batch(imgs: str, model: YOLO, out_f: str):
    t = time()
    yolo_results = model.predict(imgs, device='cuda:0', verbose=v, batch=batch_size, stream=True)
    print('setting up batch-streamed YOLO inference took', time()-t)
    t = time()
    for res in yolo_results: # 1 per img, so n in total
        if res is not None and res.masks is not None: # sth was detected
            pv_area = 0
            img_w, img_h = 640, 640 # res.orig_shape
            ppth = res.path
            mask_sum = torch.squeeze(res.masks.data).sum() # [1, 640, 640]
            pv_area = mask_sum.div(img_w*img_h) # percentage
            print(f'{time()-t},{ppth},{pv_area}\n')
            with open(out_f, 'a') as f:
                f.write(f'{ppth},{pv_area}\n')
    print('saving batch results took', time()-t)


if __name__ == '__main__':
    model = YOLO(mod)
    calculate_area_share_batch(data, model, out)
    