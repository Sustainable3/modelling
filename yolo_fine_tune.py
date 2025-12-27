from ultralytics import YOLO

model = YOLO('best.pt')

results = model.train(
    data='./auto_pv_to_fine_tunning.v4i.yolov8-obb/data.yaml',
    imgsz=640,
    epochs=150,
    lr0=0.001,
    batch=-1,

    freeze=10,               # Freezes the first 10 layers (Backbone)
                             # If results are poor, try freeze=0 to unfreeze all
    
    # --- Augmentations for Satellite/Aerial ---
    # degrees=30.0,            # Rotate images +/- 30 degrees
    # flipud=0.5,              # Vertical flip (makes sense for maps)
    # fliplr=0.5,              # Horizontal flip
    # mosaic=1.0,              # Mosaic helps with small object detection
    # scale=0.2,               # Don't zoom in/out too much (scale is consistent)

    device=0,
    cache=True,
    patience=100,
    single_cls=True,
    pretrained=True,
    project='fine_trening',
    name='finloop_1',
    workers=8
)

print(results)