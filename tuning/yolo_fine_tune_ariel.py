'''
Docstring for tuning.yolo_fine_tune_ariel

fine tuning YOLO PV model on new synthetics

MD, XII
'''
from ultralytics import YOLO

model = YOLO('final-mosaic-augmentation.pt')

results = model.train(
    data='./synth_dataset2/data.yaml',
    imgsz=640,
    batch=-1,
    device=0,
    cache=True,
    single_cls=True,
    pretrained=True,
    project='fine_trening',

    epochs=40,
    optimizer='auto',
    #lr0=0.001,
    patience=30,
    name='ariel_4',
    workers=8,
    freeze=10               # Freezes the first 10 layers (Backbone)
                             # If results are poor, try freeze=0 to unfreeze all
    
    # --- Augmentations for Satellite/Aerial ---
    # degrees=30.0,            # Rotate images +/- 30 degrees
    # flipud=0.5,              # Vertical flip (makes sense for maps)
    # fliplr=0.5,              # Horizontal flip
    # mosaic=1.0,              # Mosaic helps with small object detection
    # scale=0.2,               # Don't zoom in/out too much (scale is consistent)

)

print(results)
