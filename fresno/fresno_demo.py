"""
Docstring for fresno_demo

segmentation demo large-scale tifs
- SAHI-sliced detection with YOLO
- SAM segmentation
- composition control
- exporting segments

this code with a few awful comments and magic lines was adapted from GenAI results
and subject to very limited review

credit: 10.6084/m9.figshare.3385780

MD, XII 25
"""
import torch
import cv2
import numpy as np
import os
from ultralytics.models import SAM
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# --- CONFIGURATION ---
IMAGE_PATH = "fresno_obb/valid/images/11ska610815.tif"
YOLO_PATH = "best.pt"
# Ultralytics names them sam_b.pt (Base), sam_l.pt (Large)
SAM_MODEL_TYPE = "sam_b.pt"  # Use sam_b.pt for 4GB GPU
OUTPUT_DIR = "./output_masked_ultralytics"

SLICE_SIZE = 640
OVERLAP = 0.2
PADDING = 50 

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load YOLO (SAHI)
    print("--- Phase 1: Detection (YOLO) ---")
    yolo_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=YOLO_PATH,
        confidence_threshold=0.25,
        device=device
    )

    image_bgr = cv2.imread(IMAGE_PATH)
    h_img, w_img, _ = image_bgr.shape
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    detection_result = get_sliced_prediction(
        image_rgb,
        yolo_model,
        slice_height=SLICE_SIZE,
        slice_width=SLICE_SIZE,
        overlap_height_ratio=OVERLAP,
        overlap_width_ratio=OVERLAP,
        postprocess_type="NMS",
        postprocess_match_threshold=0.5,
        verbose=2,
        perform_standard_pred=False,
    )
    
    # Extract boxes (XYXY)
    detected_objects = detection_result.object_prediction_list
    print(f"Detected {len(detected_objects)} objects.")

    # FREE MEMORY: Delete YOLO
    del yolo_model
    torch.cuda.empty_cache()

    if not detected_objects:
        return

    # 2. Load Ultralytics SAM
    print(f"--- Phase 2: Segmentation ({SAM_MODEL_TYPE}) ---")
    # Ultralytics handles loading automatically
    sam_model = SAM(SAM_MODEL_TYPE)

    global_mask = np.zeros((h_img, w_img), dtype=bool)

    # 3. Process Crops
    for i, obj in enumerate(detected_objects):
        if i % 10 == 0: print(f"Processing object {i+1}...")

        # A. Coordinates
        x1, y1, x2, y2 = map(int, obj.bbox.to_xyxy())

        # B. Crop with padding
        crop_x1 = max(0, x1 - PADDING)
        crop_y1 = max(0, y1 - PADDING)
        crop_x2 = min(w_img, x2 + PADDING)
        crop_y2 = min(h_img, y2 + PADDING)

        crop = image_bgr[crop_y1:crop_y2, crop_x1:crop_x2] # Ultralytics likes BGR usually
        
        # C. Relative Box for Prompting
        # The object covers nearly the whole crop now, but we guide SAM exactly
        # relative box: [x1_rel, y1_rel, x2_rel, y2_rel]
        rel_box = [
            x1 - crop_x1, 
            y1 - crop_y1, 
            x2 - crop_x1, 
            y2 - crop_y1
        ]

        # D. Predict using Ultralytics syntax
        # We pass the crop and the prompt box
        results = sam_model(crop, bboxes=[rel_box], verbose=False)
        
        # E. Extract Mask
        # Ultralytics returns a Results object. 
        # result[0].masks.data is a tensor (N, H, W)
        if results[0].masks is not None:
            local_mask_tensor = results[0].masks.data[0] # Take first mask
            local_mask = local_mask_tensor.cpu().numpy().astype(bool)

            # F. Paste
            # Note: Ultralytics might resize mask output if crop size != 1024
            # We ensure it matches crop size
            if local_mask.shape != crop.shape[:2]:
                 local_mask = cv2.resize(
                     local_mask.astype(np.uint8), 
                     (crop.shape[1], crop.shape[0]), # w, h
                     interpolation=cv2.INTER_NEAREST
                 ).astype(bool)

            global_mask[crop_y1:crop_y2, crop_x1:crop_x2] |= local_mask

    # 4. Compositing
    print("--- Phase 3: Compositing ---")
    white_bg = np.full_like(image_bgr, 255)
    final_image = np.where(global_mask[..., None], image_bgr, white_bg)
    
    cv2.imwrite(os.path.join(OUTPUT_DIR, "final_result_ultralytics.png"), final_image)
    print("Done.")

if __name__ == "__main__":
    main()