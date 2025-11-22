from ultralytics import YOLO
from time import time
import torch

def sprawdz_gpu():
    print(f"Wersja PyTorch: {torch.__version__}")
    
    # 1. Sprawdzanie NVIDIA CUDA (Windows/Linux)
    if torch.cuda.is_available():
        print("\n GPU (CUDA) jest dostępne!")
        print(f"Liczba urządzeń: {torch.cuda.device_count()}")
        print(f"Nazwa obecnego GPU: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")
        
        
    # 3. Brak GPU
    else:
        print("\n GPU nie jest dostępne. Obliczenia będą wykonywane na CPU.")
        device = torch.device("cpu")
    
    print(f"Aktywne urządzenie: {device}")

if __name__ == 'main':

    models = {
        'finloop': 'best.pt',
        'ariel': 'final-mosaic-augmentation.pt',
        'pools': 'solarpanels_pools_yolov8l-p2_1024_v1.pt'
    }

    splits = {
    'pilot': ['val'],
    'rzeszow': ['val', 'test'],
    'synth': ['train', 'val', 'test']
    }

    datasets = {
    'pilot': "pilotPV_panels.v1i.yolov8-obb/data.yaml",
    'rzeszow': "rzeszowSolar panels seg.v2i.yolov8-obb/data.yaml",
    'synth': 'auto_pv_to_fine_tunning.v4i.yolov8-obb/data.yaml'
    }
    
    models = {
        'finloop': 'best.pt'
    }
    
    datasets = {
    'pilot': "pilotPV_panels.v1i.yolov8-obb/data.yaml",
    'synth': 'auto_pv_to_fine_tunning.v4i.yolov8-obb/data.yaml'
    }

    sprawdz_gpu()

    print('start')

    with open('evaluation_results.csv', 'a') as f:
        f.write('dataset,split,model,class_id,Class,Images,Instances,Box-P,Box-R,Box-F1,mAP50,mAP50-95,Mask-P,Mask-R,Mask-F1\n')
    
    for model_key, model in models.items():
        model = YOLO(model)
        suffix = ',0,0,0' if model_key != 'finloop' else ''
        for data_key, dataset in datasets.items():
            bt = 16 if model_key == 'pools' else 64
            for splt in splits[data_key]:
                t = time()
                results = model.val(data=dataset, single_cls=True, batch=bt, iou=0.7, split=splt, plots=True, project=f'runs/{data_key}_{splt}_{model_key}')
                with open('evaluation_results.csv', 'a') as f:
                    f.write(f'{data_key},{splt},{model_key},{results.to_csv(decimals=3).splitlines()[1]}{suffix}\n')
                print('done', model_key, data_key, splt, 'in', time()-t)
    
    print('end')

