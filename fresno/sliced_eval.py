import json
import os
import numpy as np
import cv2
from shapely.geometry import Polygon
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# ================= CONFIGURATION =================
# 1. Paths
JSON_GT_PATH = 'SolarArrayPolygons.json'
IMAGES_DIR = 'fresno_obb/valid/images'
MODEL_PATH = 'best.pt'

# 2. SAHI Settings (Must match your inference logic)
SLICE_SIZE = 640
OVERLAP_RATIO = 0.2
CONF_THRESHOLD = 0.25  # Confidence to count a prediction

# 3. Evaluation Settings
IOU_THRESHOLD = 0.5    # For mAP@50
# =================================================

def load_ground_truth(json_path):
    """Parses your JSON into a dict: {'image_name': [Polygon, Polygon, ...]}"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    gt_dict = {}
    for p in data.get('polygons', []):
        name = p['image_name']
        pixels = p['polygon_vertices_pixels']
        
        # Convert to Shapely Polygon
        # Flatten and reshape if necessary
        pts = np.array(pixels, dtype=np.float32).reshape(-1, 2)
        if len(pts) < 3: continue
        poly = Polygon(pts)
        
        if not poly.is_valid:
            poly = poly.buffer(0) # Fix self-intersections
            
        if name not in gt_dict:
            gt_dict[name] = []
        gt_dict[name].append(poly)
    return gt_dict

def calculate_iou(poly1, poly2):
    """Calculates IoU between two shapely polygons."""
    if not poly1.intersects(poly2):
        return 0.0
    try:
        inter_area = poly1.intersection(poly2).area
        union_area = poly1.union(poly2).area
        if union_area == 0: return 0.0
        return inter_area / union_area
    except Exception:
        return 0.0

def compute_ap(recalls, precisions):
    """Computes Average Precision (AP) using the 11-point interpolation or exact area."""
    # Append sentinel values at the end
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Integrate area under curve
    method = 'continuous' 
    if method == 'continuous':
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def evaluate_dataset():
    # 1. Load Ground Truth
    print("Loading Ground Truth...")
    gt_data = load_ground_truth(JSON_GT_PATH)
    
    # 2. Load Model
    print("Loading Model...")
    # detection_model = AutoDetectionModel(
    #     model_type='yolov8',
    #     model_path=MODEL_PATH,
    #     confidence_threshold=0.01, # Get all preds for AP curve, filter later
    #     device="cuda:0" # or 'cpu'
    # )

    detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="best.pt",  # any yolov8/yolov9/yolo11/yolo12/rt-detr det model is supported
    confidence_threshold=CONF_THRESHOLD,
    mask_threshold=IOU_THRESHOLD,
    device="cuda",  # or 'cuda:0' if GPU is available,
    # image_size=5000
    )

    all_preds = []  # List of (confidence, is_true_positive)
    total_gt_objects = 0

    print(f"Starting evaluation on {len(gt_data)} images...")

    for img_name, gt_polygons in gt_data.items():
        # Load Image
        # Check extensions
        img_path = None
        for ext in ['.jpg', '.png', '.tif']:
            p = os.path.join(IMAGES_DIR, img_name + ext)
            if os.path.exists(p):
                img_path = p
                break
        
        if not img_path:
            print(f"Skipping {img_name} (Image not found)")
            continue

        # --- RUN INFERENCE (SAHI) ---
        result = get_sliced_prediction(
            img_path,
            detection_model,
            slice_height=SLICE_SIZE,
            slice_width=SLICE_SIZE,
            overlap_height_ratio=OVERLAP_RATIO,
            overlap_width_ratio=OVERLAP_RATIO,
            verbose=2,
            perform_standard_pred=False            
        )
        
        # Convert SAHI predictions to Shapely Polygons
        pred_items = []
        for obj in result.object_prediction_list:
            if obj.score.value < 0.01: continue
            
            # SAHI returns masks or boxes. For OBB, we need the segmentation mask/polygon.
            # Usually obj.mask.segmentation gives list of points
            if hasattr(obj.mask, 'segmentation') and obj.mask.segmentation:
                # Handle potentially nested lists
                points = obj.mask.segmentation
                if isinstance(points[0], list): # flatten if needed or take exterior
                    points = points[0] # assuming single part polygon
                
                pts = np.array(points).reshape(-1, 2)
                poly = Polygon(pts)
                pred_items.append({'poly': poly, 'score': obj.score.value})
            elif obj.bbox:
                # Fallback to box if no mask
                x, y, w, h = obj.bbox.to_xywh()
                from shapely.geometry import box
                poly = box(x, y, x+w, y+h)
                pred_items.append({'poly': poly, 'score': obj.score.value})

        # --- MATCHING (Greedy Strategy) ---
        # Sort predictions by confidence (High to Low)
        pred_items.sort(key=lambda x: x['score'], reverse=True)
        
        total_gt_objects += len(gt_polygons)
        
        gt_matched = [False] * len(gt_polygons)

        for p in pred_items:
            pred_poly = p['poly']
            score = p['score']
            
            best_iou = 0
            best_gt_idx = -1
            
            # Find best matching GT that hasn't been matched yet
            for i, gt_poly in enumerate(gt_polygons):
                if gt_matched[i]:
                    continue # Already matched
                
                iou = calculate_iou(pred_poly, gt_poly)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
            
            # Determine TP or FP
            if best_iou >= IOU_THRESHOLD and best_gt_idx != -1:
                all_preds.append({'score': score, 'tp': 1})
                gt_matched[best_gt_idx] = True
            else:
                all_preds.append({'score': score, 'tp': 0})

        print(f"Processed {img_name}: {len(pred_items)} preds, {len(gt_polygons)} GTs")

    # --- CALCULATE METRICS ---
    # Sort all predictions by score
    all_preds.sort(key=lambda x: x['score'], reverse=True)

    tps = np.array([x['tp'] for x in all_preds])
    scores = np.array([x['score'] for x in all_preds])
    
    # Cumulative TPs and FPs
    cum_tps = np.cumsum(tps)
    cum_fps = np.cumsum(1 - tps)
    
    recalls = cum_tps / total_gt_objects
    precisions = cum_tps / (cum_tps + cum_fps + 1e-6)

    ap50 = compute_ap(recalls, precisions)
    
    # Metrics at specific confidence threshold (e.g. 0.25)
    # Find index closest to thresh
    idx = np.searchsorted(-scores, -CONF_THRESHOLD)
    if idx < len(scores):
        p_at_conf = precisions[idx]
        r_at_conf = recalls[idx]
        f1_at_conf = 2 * (p_at_conf * r_at_conf) / (p_at_conf + r_at_conf + 1e-6)
    else:
        p_at_conf, r_at_conf, f1_at_conf = 0, 0, 0

    print("\n" + "="*30)
    print(f"RESULTS (Huge Image Validation)")
    print(f"Images: {len(gt_data)}")
    print(f"Total GT Objects: {total_gt_objects}")
    print(f"mAP@50: {ap50:.4f}")
    print(f"At Confidence {CONF_THRESHOLD}:")
    print(f"  Precision: {p_at_conf:.4f}")
    print(f"  Recall:    {r_at_conf:.4f}")
    print(f"  F1 Score:  {f1_at_conf:.4f}")
    print("="*30)

if __name__ == "__main__":
    evaluate_dataset()