"""combined_PV_eval.ipynb

# PV combined evaluation
"""

"""source (Huggingface): finloop/yolov8s-seg-solar-panels (aka Rzeszów model)

source (Huggingface): https://huggingface.co/spaces/ArielDrabkin/Solar-Panel-Detector

source (Huggingface): andrewgray11/autotrain-solar-panel-object-detection-50559120777

XI 25

*MD*
"""

from ultralytics import YOLO
from time import time
import torch

def sprawdz_gpu():
    print(f"Wersja PyTorch: {torch.__version__}")
    
    if torch.cuda.is_available():
        print("\n GPU (CUDA) jest dostępne!")
        print(f"Liczba urządzeń: {torch.cuda.device_count()}")
        print(f"Nazwa obecnego GPU: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")
        
    else:
        print("\n GPU nie jest dostępne. Obliczenia będą wykonywane na CPU.")
        device = torch.device("cpu")
    
    print(f"Aktywne urządzenie: {device}")

if __name__ == '__main__':

    datasets = {
        'pilot': "pilotPV_panels.v1i.yolov8-obb/data.yaml",
        # 'rzeszow': "rzeszowSolar panels seg.v2i.yolov8-obb/data.yaml",
        'synth': 'auto_pv_to_fine_tunning.v4i.yolov8-obb/data.yaml'
    }

    splits = {
        'pilot': ['val'], # in yaml test=val or so, adjustable
        'rzeszow': ['val', 'test'],
        'synth': ['train', 'val', 'test']
    }

    models = {
        'finloop': 'best.pt',
        'ariel': 'final-mosaic-augmentation.pt',
        'pools': 'solarpanels_pools_yolov8l-p2_1024_v1.pt'
    }

    sprawdz_gpu()

    print('start')

    with open('evaluation_results.csv', 'w') as f:
        f.write('dataset,split,model,Class,Images,Instances,Box-P,Box-R,Box-F1,mAP50,mAP50-95,Mask-P,Mask-R,Mask-F1,t\n')

    for data_key, dataset in datasets.items():
        for splt in splits[data_key]:
            for model_key, model in models.items():
                bt = 16 if model_key == 'pools' else 64
                model = YOLO(model)
                suffix = ',0,0,0' if model_key != 'finloop' else ''
                t = time()
                results = model.val(data=dataset, single_cls=True, batch=bt, iou=0.7, split=splt, plots=True, project=f'runs/{data_key}_{splt}_{model_key}')
                t = time()-t
                with open('evaluation_results.csv', 'a') as f:
                    f.write(f'{data_key},{splt},{model_key},{results.to_csv(decimals=3).splitlines()[1]}{suffix},{t}\n')
                print('done', model_key, data_key, splt, 'in', t)

    print('end')