'''
Docstring for fresno.fine_tune

fine tuning YOLO PV model on old synthetics

this code with a few awful comments and magic lines was adapted from GenAI results
and subject to limited review

MD, XII

'''
from ultralytics import YOLO

# 1. Load your pre-trained model
#    If you have a generic OBB model, use 'yolov8n-obb.pt'
#    If you have a custom one, use 'path/to/best.pt'
model = YOLO('best.pt')

# 2. Train with Fine-Tuning Hyperparameters
results = model.train(
    # --- Data ---
    data='./auto_pv_to_fine_tunning.v4i.yolov8-obb/data.yaml',        # Points to your tiled dataset
    imgsz=640,               # Match your tiling script output
    
    # --- Fine-Tuning Logic ---
    epochs=150,               # Shorter training cycle
    lr0=0.001,               # Low learning rate to preserve pre-trained weights
    batch=-1,                # Adjust based on your GPU VRAM
    
    # --- Freezing ---
    freeze=10,               # Freezes the first 10 layers (Backbone)
                             # If results are poor, try freeze=0 to unfreeze all
    
    # --- Augmentations for Satellite/Aerial ---
    # degrees=30.0,            # Rotate images +/- 30 degrees
    # flipud=0.5,              # Vertical flip (makes sense for maps)
    # fliplr=0.5,              # Horizontal flip
    # mosaic=1.0,              # Mosaic helps with small object detection
    # scale=0.2,               # Don't zoom in/out too much (scale is consistent)
    
    # --- Hardware ---
    device=0,                # GPU index
    cache=True,
    single_cls=True,
    pretrained=True,
    project='fine_trening',
    name='finloop_1',
    workers=8                # Data loading threads
)

print(results)