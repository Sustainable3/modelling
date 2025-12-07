import torch
import cv2
import numpy as np
import os
import csv
from glob import glob
from tqdm import tqdm
from ultralytics.models import SAM
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# --- CONFIGURATION ---
IMAGES_DIR = "fresno_obb/valid/images/"        # Input Directory
OUTPUT_DIR = "./output_masked_ultralytics2"      # Output Directory for images
CSV_PATH = "area_stats2.csv"         # Output CSV for statistics

YOLO_PATH = "best.pt"
SAM_MODEL_TYPE = "sam_b.pt"         # sam_l.pt

# SAHI Config
SLICE_SIZE = 640
OVERLAP = 0.2
PADDING = 50                        # Context padding for SAM crops

def main():
    # Setup
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image_files = glob(os.path.join(IMAGES_DIR, "*.tif"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Found {len(image_files)} images to process.")

    # Container to store detection results temporarily
    # Format: { "image_path": [[x1, y1, x2, y2], ...] }
    detection_metadata = {}

    # ==========================================
    # PHASE 1: Detection (YOLO + SAHI)
    # ==========================================
    print("\n" + "="*40)
    print("PHASE 1: Detection (YOLO)")
    print("="*40)
    
    yolo_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=YOLO_PATH,
        confidence_threshold=0.45,
        device=device
    )

    for img_path in tqdm(image_files, desc="Detecting"):
        image_bgr = cv2.imread(img_path)
        if image_bgr is None: continue
        
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Run SAHI
        result = get_sliced_prediction(
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

        # Store Boxes
        boxes = []
        for obj in result.object_prediction_list:
            boxes.append(obj.bbox.to_xyxy()) # [x1, y1, x2, y2]
            
        detection_metadata[img_path] = boxes

    # CLEANUP PHASE 1
    del yolo_model
    torch.cuda.empty_cache()
    print("Detection complete. YOLO unloaded.", len(boxes), 'boxes detected')

    # ==========================================
    # PHASE 2: Segmentation (SAM) & Area Calc
    # ==========================================
    print("\n" + "="*40)
    print(f"PHASE 2: Segmentation ({SAM_MODEL_TYPE})")
    print("="*40)

    # Initialize CSV
    with open(CSV_PATH, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Total_Pixels_Image", "Object_Pixels", "Object_Percentage", "Num_Objects"])

    # Load SAM
    sam_model = SAM(SAM_MODEL_TYPE)
    # Note: Ultralytics SAM loads to GPU automatically on first inference usually, 
    # but we can force empty cache just in case.
    
    for img_path, boxes in tqdm(detection_metadata.items(), desc="Segmenting"):
        if not boxes:
            # Log empty image and skip
            write_stats(img_path, 0, 0, 0)
            continue

        # Load Image (Again)
        image_bgr = cv2.imread(img_path)
        h_img, w_img, _ = image_bgr.shape
        
        global_mask = np.zeros((h_img, w_img), dtype=bool)

        # Process Every Detected Box
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            # Crop with Padding
            crop_x1 = max(0, x1 - PADDING)
            crop_y1 = max(0, y1 - PADDING)
            crop_x2 = min(w_img, x2 + PADDING)
            crop_y2 = min(h_img, y2 + PADDING)

            crop = image_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
            
            if crop.size == 0: continue

            # Relative Box for SAM Prompt
            rel_box = [
                x1 - crop_x1, 
                y1 - crop_y1, 
                x2 - crop_x1, 
                y2 - crop_y1
            ]

            # Run SAM Inference
            results = sam_model(crop, bboxes=[rel_box], verbose=False)

            if results[0].masks is not None:
                # Extract Mask
                local_mask = results[0].masks.data[0].cpu().numpy().astype(bool)

                # Resize check (in case SAM output differs from Crop input)
                if local_mask.shape != crop.shape[:2]:
                    local_mask = cv2.resize(
                        local_mask.astype(np.uint8), 
                        (crop.shape[1], crop.shape[0]), 
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)

                # Paste into Global Mask
                global_mask[crop_y1:crop_y2, crop_x1:crop_x2] |= local_mask

        # --- CALCULATIONS ---
        total_pixels = h_img * w_img
        object_pixels = np.count_nonzero(global_mask)
        percentage = (object_pixels / total_pixels) * 100

        # Save to CSV
        write_stats(img_path, total_pixels, object_pixels, percentage, len(boxes))

        # --- SAVE MASKED IMAGE ---
        # Create white background
        white_bg = np.full_like(image_bgr, 255)
        # Composite
        final_image = np.where(global_mask[..., None], image_bgr, white_bg)
        
        filename = os.path.basename(img_path)
        save_path = os.path.join(OUTPUT_DIR, f"masked_{filename}")
        cv2.imwrite(save_path, final_image)

    print(f"\nProcessing complete. Stats saved to {CSV_PATH}")

def write_stats(img_path, total, obj_px, pct, num_objs=0):
    with open(CSV_PATH, mode='a', newline='') as f:
        writer = csv.writer(f)
        filename = os.path.basename(img_path)
        writer.writerow([filename, total, obj_px, f"{pct:.4f}", num_objs])

if __name__ == "__main__":
    main()