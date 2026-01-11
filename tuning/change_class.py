'''
change the name given to the class in a pretrained model

actually, official YOLOs are pretrained on COCO
which has class 0 as human
so subject to change too?
https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml

Ultralytics' genAI

I 26, MD
'''
from ultralytics import YOLO

# Load your model
model = YOLO("ariel.pt")

# Update names (index: "new_name")
model.model.names = {0: "pv"}

# Save the updated model
model.save("ariel.pt")
