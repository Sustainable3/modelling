# %% [markdown]
# # PV segmentation

# %% [markdown]
# Comparison between models (YOLOdetect + SAM)

# %% [markdown]
# source (Huggingface): finloop/yolov8s-seg-solar-panels (aka Rzeszów model)
# 
# source (Huggingface): https://huggingface.co/spaces/ArielDrabkin/Solar-Panel-Detector
# 
# source (Huggingface): andrewgray11/autotrain-solar-panel-object-detection-50559120777
# 
# credits: https://blog.roboflow.com/how-to-use-yolov8-with-sam/ (Roboflow)

# %% [markdown]
# XI 25
# 
# *MD*

# %% [markdown]
# ## libs

# %%
# %pip install numpy
# %pip install pandas
# %pip install ultralytics

# %%
#import matplotlib.pyplot as plt
#import cv2
from ultralytics import YOLO, SAM
import torch
from time import time

# %%
dev = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
dev

# %% [markdown]
# ## defs

# %% [markdown]
# ### variants

# %%
sam = SAM("sam_b.pt")

# %%
models = {
    'finloop': 'best.pt'
}

# %%
datasets = {
    'pilot': 'pilotPV_panels.v1i.yolov8-obb/test/images/*.jpg',
    #'rzeszow_test': 'rzeszowSolar panels seg.v2i.yolov8-obb/test/images/*.jpg',
    # 'rzeszow_train': 'rzeszowSolar panels seg.v2i.yolov8-obb/train/images/*.jpg',
    #'rzeszow_valid': 'rzeszowSolar panels seg.v2i.yolov8-obb/valid/images/*.jpg',
    'synth_test': 'auto_pv_to_fine_tunning.v4i.yolov8-obb/test/images/*.jpg',
    'synth_train': 'auto_pv_to_fine_tunning.v4i.yolov8-obb/train/images/*.jpg',
    'synth_valid': 'auto_pv_to_fine_tunning.v4i.yolov8-obb/valid/images/*.jpg'
}

# %% [markdown]
# ### segment analysis

# %%
def sum_pv_segments(pth, model, nazwa="no_info_run", print_info=False, disp_img=False, display_coef=100):
    pv_area = 0
    yolo_results = model(pth, save=print_info, name=nazwa, stream=True, device=dev, verbose=print_info)
    for i, res in enumerate(yolo_results):
        dsp = disp_img and i % display_coef == 0 # limit
        ppth = res.path # nicely conveyed
        img_w, img_h = res.orig_shape # although YOLO reshapes img when necessary, SAM masks match org img
        if dsp:
            image = cv2.cvtColor(cv2.imread(ppth), cv2.COLOR_BGR2RGB)
            image = torch.tensor(image, device=dev)
        if res is not None and res.boxes is not None and res.boxes.xyxy is not None and len(res.boxes.xyxy > 0): # null-len res.boxes.xyxy for no PV
            sam_results = sam.predict(source=ppth, bboxes=res.boxes.xyxy)
            if sam_results is not None and sam_results[0] is not None and sam_results[0].masks is not None and sam_results[0].masks.data is not None:
                binary_mask = torch.where(sam_results[0].masks.data == True, 1, 0)
                mask_sum = binary_mask.sum(axis=0).data
                mask_sum_damped = torch.where(mask_sum >= 1, 1, 0) # actually some pxs covered multiple times - same object
                if dsp:
                    bcg_white = torch.ones_like(image)*255
                    new_image = bcg_white * (1 - mask_sum_damped[..., torch.newaxis]) + image * mask_sum_damped[..., torch.newaxis]
                    plt.imshow(new_image.reshape((img_w, img_h, 3)).cpu())
                    plt.title(f"Masked PVs in {ppth[ppth.rfind('/'):]}")
                    plt.axis('off')
                    plt.show()
                    # cv2.imwrite('c.png', new_image.reshape((img_w, img_h, 3)).cpu().numpy())
                pv_area += mask_sum_damped.sum().div(img_w*img_h) # percentage
                # print('mask sums', mask_sum.sum(), mask_sum_damped.sum())
                if print_info:
                    print(i, pv_area.item())
        if dsp:
            plt.imshow(image.cpu())
            plt.title(f"base img {ppth[ppth.rfind('/'):]}")
            plt.axis('off')
            plt.show()
    return pv_area

# %%
def sum_pv_segments_sam(pth, model, nazwa="no_info_run", print_info=False, disp_img=False, display_coef=100):
    pv_area = 0
    yolo_results = model(pth, save=print_info, name=nazwa, stream=True, device=dev, verbose=print_info)
    for i, res in enumerate(yolo_results):
        dsp = disp_img and i % display_coef == 0 # limit
        ppth = res.path # nicely conveyed
        img_w, img_h = res.orig_shape # although YOLO reshapes img when necessary, SAM masks match org img
        if dsp:
            image = cv2.cvtColor(cv2.imread(ppth), cv2.COLOR_BGR2RGB)
            image = torch.tensor(image, device=dev)
        if res is not None and res.boxes is not None and res.boxes.xyxy is not None and len(res.boxes.xyxy > 0): # null-len res.boxes.xyxy for no PV
            sam_results = sam.predict(source=ppth, bboxes=res.boxes.xyxy)
            if sam_results is not None and sam_results[0] is not None and sam_results[0].masks is not None and sam_results[0].masks.data is not None:
                binary_mask = torch.where(sam_results[0].masks.data == True, 1, 0)
                mask_sum = binary_mask.sum(axis=0).data
                mask_sum_damped = torch.where(mask_sum >= 1, 1, 0) # actually some pxs covered multiple times - same object
                if dsp:
                    bcg_white = torch.ones_like(image)*255
                    new_image = bcg_white * (1 - mask_sum_damped[..., torch.newaxis]) + image * mask_sum_damped[..., torch.newaxis]
                    plt.imshow(new_image.reshape((img_w, img_h, 3)).cpu())
                    plt.title(f"Masked PVs in {ppth[ppth.rfind('/'):]}")
                    plt.axis('off')
                    plt.show()
                    # cv2.imwrite('c.png', new_image.reshape((img_w, img_h, 3)).cpu().numpy())
                pv_area += mask_sum_damped.sum().div(img_w*img_h) # percentage
                # print('mask sums', mask_sum.sum(), mask_sum_damped.sum())
                if print_info:
                    print(i, pv_area.item())
        if dsp:
            plt.imshow(image.cpu())
            plt.title(f"base img {ppth[ppth.rfind('/'):]}")
            plt.axis('off')
            plt.show()
    return pv_area

# %% [markdown]
# ## run

# %%
with open('finloop_comparison.csv', 'a') as f:
    f.write('data_key,mode,area,time\n')
print('full yolo segment')
for model_key, model in models.items():
    model = YOLO(model)
    for data_key, dataset in datasets.items():
        t = time()
        area = sum_pv_segments(dataset, model=model, nazwa=data_key, print_info=False)
        with open('finloop_comparison.csv', 'a') as f:
            f.write(f'{data_key},segment,{area},{time()-t}\n')
            
for model_key, model in models.items():
    model = YOLO(model)
    print('yolo with sam')
    for data_key, dataset in datasets.items():
        t = time()
        area = sum_pv_segments_sam(dataset, model=model, nazwa=data_key, print_info=False)
        with open('finloop_comparison.csv', 'a') as f:
            f.write(f'{data_key},SAM,{area},{time()-t}\n')


