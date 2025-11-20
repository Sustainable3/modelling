from ultralytics import YOLO
from time import time

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

    print('start')

    with open('evaluation_results.csv', 'a') as f:
        f.write('dataset,split,model,class_id,Class,Images,Instances,Box-P,Box-R,Box-F1,mAP50,mAP50-95,Mask-P,Mask-R,Mask-F1\n')
    
    for model_key, model in models.items():
        model = YOLO(model)
        for data_key, dataset in datasets.items():
            bt = 16 if model_key == 'pools' else 64
            for splt in splits[data_key]:
                t = time()
                results = model.val(data=dataset, single_cls=True, batch=bt, iou=0.7, split=splt, plots=True, project=f'runs/{data_key}_{splt}_{model_key}')
                with open('evaluation_results.csv', 'a') as f:
                    f.write(f'{data_key},{splt},{model_key},{results.to_csv(decimals=3).splitlines()[1]}\n')
                print('done', model_key, data_key, splt, 'in', time()-t)
    
    print('end')

