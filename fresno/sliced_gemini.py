import os
import cv2
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
MODEL_PATH = "best.pt"           # Path to your YOLOv8/v11 model
IMAGES_DIR = "fresno_obb/valid/images"     # Path to 5000x5000 TIF images
IMAGES_DIR = "fresno_seg/valid/images"     # Path to 5000x5000 TIF images
JSON_GT_PATH = "SolarArrayPolygons.json" # Path to the JSON file
IMG_EXT = ".tif"

SLICE_SIZE = 640                 # Size of the chips fed to the model
OVERLAP_RATIO = 0.2              # Overlap between slices
IOU_THRESHOLD = 0.5              # IoU for NMS stitching
CONF_THRESHOLD = 0.25            # Confidence threshold

# Select Metrics to Compute
COMPUTE_BOX = True
COMPUTE_MASK = False

class GroundTruthLoader:
    def __init__(self, json_path):
        print(f"Loading Ground Truth JSON from {json_path}...")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Group polygons by image_name for fast lookup
        # Structure: { "image_name_no_ext": [polygon_entry, ...] }
        self.gt_map = defaultdict(list)
        
        # Check if root is list or dict based on user snippet
        polygons = data.get("polygons", []) if isinstance(data, dict) else data
        
        for poly in polygons:
            img_name = poly.get("image_name")
            if img_name:
                self.gt_map[img_name].append(poly)
                
        print(f"Loaded {len(polygons)} polygons across {len(self.gt_map)} images.")

    def get_labels(self, img_name_no_ext, img_height, img_width):
        """
        Returns:
            boxes: Tensor [N, 4] (xyxy)
            labels: Tensor [N]
            masks: List of boolean numpy arrays
        """
        if img_name_no_ext not in self.gt_map:
            return torch.tensor([]), torch.tensor([]), []

        poly_list = self.gt_map[img_name_no_ext]
        
        boxes = []
        classes = []
        masks = []

        for item in poly_list:
            # Extract vertices: [[x1, y1], [x2, y2], ...]
            # Note: JSON lists usually come as floats, we cast to int for cv2 drawing
            vertices = np.array(item["polygon_vertices_pixels"], dtype=np.int32)
            
            if vertices.size == 0:
                continue

            # 1. Generate Box (XYXY)
            x_min = np.min(vertices[:, 0])
            y_min = np.min(vertices[:, 1])
            x_max = np.max(vertices[:, 0])
            y_max = np.max(vertices[:, 1])
            
            # Clip to image boundaries to avoid errors
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img_width, x_max)
            y_max = min(img_height, y_max)

            boxes.append([x_min, y_min, x_max, y_max])
            
            # 2. Assign Class
            # Assuming single class (e.g., building) since not specified in JSON
            # Change '0' to a lookup if 'type' exists in your JSON
            classes.append(0) 

            # 3. Generate Mask
            if COMPUTE_MASK:
                mask = np.zeros((img_height, img_width), dtype=np.uint8)
                cv2.fillPoly(mask, [vertices], 1)
                masks.append(mask.astype(bool))

        if not boxes:
            return torch.tensor([]), torch.tensor([]), []

        return (
            torch.tensor(boxes, dtype=torch.float32), 
            torch.tensor(classes, dtype=torch.int32), 
            masks
        )

def main():
    # 1. Initialize SAHI Model
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=MODEL_PATH,
        confidence_threshold=CONF_THRESHOLD,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # 2. Initialize GT Loader
    gt_loader = GroundTruthLoader(JSON_GT_PATH)

    # 3. Initialize Evaluator
    metric = MeanAveragePrecision(
        box_format="xyxy", 
        iou_type="segm" if COMPUTE_MASK else "bbox",
        extended_summary=True
    )

    image_files = glob(os.path.join(IMAGES_DIR, f"*{IMG_EXT}"))
    print(f"Found {len(image_files)} images for evaluation.")

    for img_path in tqdm(image_files, desc="Evaluating"):
        # -- A. Load Image --
        # Use filename without extension to match JSON keys
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            print(f"Warning: Could not read {img_path}")
            continue
            
        h, w, _ = image_bgr.shape
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # -- B. Load Ground Truth from JSON --
        gt_boxes, gt_labels, gt_masks = gt_loader.get_labels(base_name, h, w)
        
        if gt_boxes.shape[0] == 0:
            # Decide if you want to skip or count as empty (False Positives only)
            # Generally for metrics, we want to include images even if empty
            pass

        target = dict(
            boxes=gt_boxes.to("cpu"),
            labels=gt_labels.to("cpu"),
        )
        if COMPUTE_MASK and gt_masks:
            target["masks"] = torch.tensor(np.stack(gt_masks), dtype=torch.bool)

        # -- C. Run Inference (SAHI) --
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
            perform_standard_pred=False
        )

        # -- D. Format Predictions --
        pred_boxes = []
        pred_scores = []
        pred_labels = []
        pred_masks = []

        for obj in result.object_prediction_list:
            pred_boxes.append(obj.bbox.to_xyxy()) 
            pred_scores.append(obj.score.value)
            pred_labels.append(obj.category.id)
            
            if COMPUTE_MASK and obj.mask:
                if hasattr(obj.mask, 'bool_mask'):
                    pred_masks.append(obj.mask.bool_mask)
                else:
                    m = np.zeros((h, w), dtype=np.uint8)
                    pts = np.array(obj.mask.segmentation, dtype=np.int32)
                    cv2.fillPoly(m, [pts], 1)
                    pred_masks.append(m.astype(bool))

        if len(pred_boxes) == 0:
            preds = dict(
                boxes=torch.tensor([], device="cpu"),
                scores=torch.tensor([], device="cpu"),
                labels=torch.tensor([], device="cpu")
            )
            if COMPUTE_MASK:
                 preds["masks"] = torch.tensor([], dtype=torch.bool, device="cpu")
        else:
            preds = dict(
                boxes=torch.tensor(pred_boxes, dtype=torch.float32),
                scores=torch.tensor(pred_scores, dtype=torch.float32),
                labels=torch.tensor(pred_labels, dtype=torch.int32)
            )
            if COMPUTE_MASK and pred_masks:
                 preds["masks"] = torch.tensor(np.stack(pred_masks), dtype=torch.bool)

        # -- E. Update Metric --
        metric.update([preds], [target])

    # 4. Compute Final Metrics
    print("Computing final metrics...")
    results = metric.compute()
    
    # Pretty Print Results
    # TorchMetrics returns a dict with tensors
    print("\n" + "=" * 40)
    print("   YOLO + SAHI Evaluation Results")
    print("=" * 40)
    
    def print_metric(name, key):
        if key in results:
            val = results[key].item()
            print(f"{name:<15} : {val:.4f}")
            
    print_metric("mAP 50-95", 'map')
    print_metric("mAP 50", 'map_50')
    print_metric("mAP 75", 'map_75')
    print("-" * 40)
    print_metric("Recall (Small)", 'map_small')
    print_metric("Recall (Medium)", 'map_medium')
    print_metric("Recall (Large)", 'map_large')
    print("=" * 40)

if __name__ == "__main__":
    main()