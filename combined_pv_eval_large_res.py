"""
Docstring for combined_pv_eval_large_res

Fresno data

evaluating prediction quality
- different models - obb-detect mode
- large imagery - slicing with SAHI
- extracting ground truth from an extensive json
- saving metric values to csv
- 2 imgs for now

TODO:
- metrics - add some sense
- thresholds
- imgs - download more

this code with a few awful comments and magic lines was adapted from GenAI results
and subject to very limited review

credit: 10.6084/m9.figshare.3385780

MD, XII 25
"""
import os
import cv2
import csv
import json
import torch
import numpy as np
from glob import glob
from tqdm import tqdm
from collections import defaultdict
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# --- CONFIGURATION ---
MODELS_DIR = ""  # Directory containing .pt files
IMAGES_DIR = "fresno_obb/valid/images"
JSON_GT_PATH = "SolarArrayPolygons.json"
OUTPUT_CSV = "fresno_results2.csv"
IMG_EXT = ".tif"

# SAHI & Inference Config
SLICE_SIZE = 640
OVERLAP_RATIO = 0.2
IOU_THRESHOLD = 0.5  # For NMS stitching
CONF_THRESHOLD = 0.25

# Metrics Config
COMPUTE_MASK = False  # Set False if models are Box-only

# --- GROUND TRUTH LOADER (Reusable Class) ---
class GroundTruthLoader:
    def __init__(self, json_path):
        print(f"Loading Ground Truth JSON from {json_path}...")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        self.gt_map = defaultdict(list)
        polygons = data.get("polygons", []) if isinstance(data, dict) else data
        
        for poly in polygons:
            img_name = poly.get("image_name")
            if img_name:
                self.gt_map[img_name].append(poly)
        print(f"Loaded {len(polygons)} polygons across {len(self.gt_map)} images.")

    def get_labels(self, img_name_no_ext, img_height, img_width):
        if img_name_no_ext not in self.gt_map:
            return torch.tensor([]), torch.tensor([]), []

        poly_list = self.gt_map[img_name_no_ext]
        boxes, classes, masks = [], [], []

        for item in poly_list:
            # FIX: Ensure int32 for OpenCV
            vertices = np.array(item["polygon_vertices_pixels"], dtype=np.int32)
            if vertices.size == 0: continue
            
            # Reshape for safety (N, 2)
            vertices = vertices.reshape((-1, 2))

            # 1. Box
            x_min, y_min = np.min(vertices, axis=0)
            x_max, y_max = np.max(vertices, axis=0)
            # Clip
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(img_width, x_max), min(img_height, y_max)
            boxes.append([x_min, y_min, x_max, y_max])
            
            # 2. Class (Default 0)
            classes.append(0) 

            # 3. Mask
            if COMPUTE_MASK:
                mask = np.zeros((img_height, img_width), dtype=np.uint8)
                # FIX: Wrap vertices in list [vertices]
                cv2.fillPoly(mask, [vertices], 1)
                masks.append(mask.astype(bool))

        if not boxes:
            return torch.tensor([]), torch.tensor([]), []

        return (
            torch.tensor(boxes, dtype=torch.float32), 
            torch.tensor(classes, dtype=torch.int32), 
            masks
        )

# --- SINGLE MODEL EVALUATOR ---
def evaluate_model(model_path, gt_loader, image_files):
    """
    Runs SAHI inference and computes metrics for a single model.
    """
    model_name = os.path.basename(model_path)
    
    # Load Model
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=model_path,
        confidence_threshold=CONF_THRESHOLD,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Setup Metric
    metric = MeanAveragePrecision(
        box_format="xyxy", 
        iou_type="segm" if COMPUTE_MASK else "bbox",
        extended_summary=False
    )

    count_files = 0
    count_detected = 0
    count_expected = 0

    # Inference Loop
    for img_path in tqdm(image_files, desc=f"Eval {model_name}", leave=False):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Load Image
        image_bgr = cv2.imread(img_path)
        if image_bgr is None: continue
        h, w, _ = image_bgr.shape
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        count_files += 1

        # Load GT
        gt_boxes, gt_labels, gt_masks = gt_loader.get_labels(base_name, h, w)

        count_expected += len(gt_boxes)
        
        target = dict(boxes=gt_boxes.to("cpu"), labels=gt_labels.to("cpu"))
        if COMPUTE_MASK and gt_masks:
            target["masks"] = torch.tensor(np.stack(gt_masks), dtype=torch.bool)

        # SAHI Inference
        result = get_sliced_prediction(
            image_rgb,
            detection_model,
            slice_height=SLICE_SIZE,
            slice_width=SLICE_SIZE,
            overlap_height_ratio=OVERLAP_RATIO,
            overlap_width_ratio=OVERLAP_RATIO,
            postprocess_type="NMS",
            postprocess_match_metric="IOS", 
            postprocess_match_threshold=IOU_THRESHOLD,
            verbose=2,
            perform_standard_pred=False,
        )

        count_detected += len(result.object_prediction_list)

        # Format Preds
        pred_boxes, pred_scores, pred_labels, pred_masks = [], [], [], []
        
        for obj in result.object_prediction_list:
            pred_boxes.append(obj.bbox.to_xyxy()) 
            pred_scores.append(obj.score.value)
            pred_labels.append(obj.category.id)
            
            if COMPUTE_MASK and obj.mask:
                # Handle SAHI mask formats
                if hasattr(obj.mask, 'bool_mask'):
                    pred_masks.append(obj.mask.bool_mask)
                else:
                    # FIX: Correct reshape and int32 cast
                    m = np.zeros((h, w), dtype=np.uint8)
                    pts = np.array(obj.mask.segmentation, dtype=np.int32).reshape((-1, 2))
                    cv2.fillPoly(m, [pts], 1)
                    pred_masks.append(m.astype(bool))

        if len(pred_boxes) == 0:
            preds = dict(
                boxes=torch.tensor([], device="cpu"),
                scores=torch.tensor([], device="cpu"),
                labels=torch.tensor([], device="cpu")
            )
            if COMPUTE_MASK: preds["masks"] = torch.tensor([], dtype=torch.bool, device="cpu")
        else:
            preds = dict(
                boxes=torch.tensor(pred_boxes, dtype=torch.float32),
                scores=torch.tensor(pred_scores, dtype=torch.float32),
                labels=torch.tensor(pred_labels, dtype=torch.int32)
            )
            if COMPUTE_MASK and pred_masks:
                 preds["masks"] = torch.tensor(np.stack(pred_masks), dtype=torch.bool)

        metric.update([preds], [target])

    # Compute & Clean up
    results = metric.compute()
    
    # Unload model to free GPU memory
    del detection_model
    torch.cuda.empty_cache()
    
    stats = {
        "files_processed": count_files,
        "instances_detected": count_detected,
        "instances_expected": count_expected
    }
    return results, stats

# --- MAIN CONTROLLER ---
def main():
    # 1. Setup
    model_files = glob(os.path.join(MODELS_DIR, "*.pt"))
    image_files = glob(os.path.join(IMAGES_DIR, f"*{IMG_EXT}"))
    gt_loader = GroundTruthLoader(JSON_GT_PATH)
    
    print(f"Found {len(model_files)} models and {len(image_files)} images.")

    # 2. Prepare CSV
    # COCO Metrics keys
    fieldnames = [
        "model_name", 
        "files_processed",      # <--- New
        "instances_expected",   # <--- New
        "instances_detected",   # <--- New
        "map", "map_50", "map_75", 
        "map_small", "map_medium", "map_large",
        "mar_1", "mar_10", "mar_100", 
        "mar_small", "mar_medium", "mar_large"
    ]
    
    # Initialize file with headers
    with open(OUTPUT_CSV, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    # 3. Iterate Models
    for model_path in model_files:
        model_name = os.path.basename(model_path)
        print(f"\n--- Processing: {model_name} ---")
        
        try:
            # Run Evaluation
            metrics, stats = evaluate_model(model_path, gt_loader, image_files)
            
            # Format row for CSV
            row = {"model_name": model_name}
            row.update(stats)
            for k, v in metrics.items():
                if k in fieldnames:
                    row[k] = f"{v.item():.4f}" # Convert tensor to float string
            
            # Write immediately
            with open(OUTPUT_CSV, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row)
                
            print(f"Saved: {model_name} | Found {stats['instances_detected']} / {stats['instances_expected']} Objects")
            
        except Exception as e:
            print(f"!!! Error evaluating {model_name}: {e}")
            # Optional: write error to CSV or log file
            continue

    print(f"\nAll Done. Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()