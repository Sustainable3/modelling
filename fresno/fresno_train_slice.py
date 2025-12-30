'''
Docstring for fresno.fresno_train_slice

from Fresno

the proper img slicer for dataset creation

takes Fresno .tifs and the .json
cuts imgs into small pieces
and transforms polygon labels from .json into small pieces

this code with a few awful comments and magic lines was adapted from GenAI results
and subject to limited review

credit: 10.6084/m9.figshare.3385780
MD, XII

'''
import json
import os
import cv2
import numpy as np
import shutil
import random
from shapely.geometry import Polygon, box
from shapely.affinity import translate

# ================= CONFIGURATION =================
# 1. Inputs
JSON_FILE = 'SolarArrayPolygons.json' # polygons
IMAGES_DIR = './fresno_obb/valid/images' # just pure imgs needed

# 2. Output
OUTPUT_DIR = './yolo_dataset_ready'

# 3. Tiling Settings
TILE_SIZE = 640                  # Target size for model
OVERLAP = 0.2                    # 20% overlap
STRIDE = int(TILE_SIZE * (1 - OVERLAP))

# 4. Filter Settings
MIN_AREA_RATIO = 0.3             # Discard object if < 30% is visible in tile
VAL_SPLIT = 0.2                  # 20% validation data
CLASS_ID = 0
# =================================================

def ensure_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)  # Clean start
    os.makedirs(path)

def make_yolo_dirs(base_path):
    for split in ['train', 'val']:
        os.makedirs(os.path.join(base_path, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_path, split, 'labels'), exist_ok=True)

def parse_polygons_from_json(json_path):
    """Parses JSON and returns dict: {'img_name': [ShapelyPoly, ...]}"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    grouped = {}
    for p in data.get('polygons', []):
        img_name = p['image_name']
        raw_pts = p['polygon_vertices_pixels']
        
        # Robust conversion to Nx2 array
        pts = np.array(raw_pts, dtype=np.float32).reshape(-1, 2)
        
        if len(pts) >= 3:
            poly = Polygon(pts)
            if not poly.is_valid:
                poly = poly.buffer(0) # Fix self-intersections
                
            if img_name not in grouped:
                grouped[img_name] = []
            grouped[img_name].append(poly)
            
    return grouped

def get_yolo_obb_string(shapely_poly, tile_w, tile_h):
    """
    Converts a Shapely Polygon (in tile pixels) to YOLO OBB String.
    Format: class x1 y1 x2 y2 x3 y3 x4 y4 (Normalized 0-1)
    """
    # 1. Calculate Minimum Rotated Rectangle
    rect = shapely_poly.minimum_rotated_rectangle
    
    # 2. Extract 4 corners
    # shapely returns 5 points (start=end), slice to 4
    try:
        x, y = rect.exterior.coords.xy
        coords = list(zip(x, y))[:4]
    except AttributeError:
        # Fallback if rect is not a polygon (e.g. point/line due to weird clip)
        return None

    # 3. Normalize
    norm_vals = []
    for (px, py) in coords:
        # Clip to ensure 0.0-1.0 (handling float precision errors)
        nx = max(0.0, min(1.0, px / tile_w))
        ny = max(0.0, min(1.0, py / tile_h))
        norm_vals.extend([nx, ny])
        
    return f"{CLASS_ID} " + " ".join([f"{v:.6f}" for v in norm_vals])

def verify_and_draw(img_path, label_path, save_path):
    """Debug function to draw the created labels onto the image."""
    img = cv2.imread(img_path)
    if img is None: return
    h, w = img.shape[:2]
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        coords = [float(x) for x in parts[1:]]
        
        # Denormalize
        pts = []
        for i in range(0, len(coords), 2):
            px = int(coords[i] * w)
            py = int(coords[i+1] * h)
            pts.append([px, py])
            
        # Draw Polygon (Green)
        pts_np = np.array(pts, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts_np], True, (0, 255, 0), 2)
        
        # Draw Start Point (Red Dot) to visualize orientation
        cv2.circle(img, tuple(pts[0]), 3, (0, 0, 255), -1)

    cv2.imwrite(save_path, img)

def process_dataset():
    print("--- 1. Initialization ---")
    if os.path.exists(OUTPUT_DIR):
        print(f"Removing existing {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)
    make_yolo_dirs(OUTPUT_DIR)
    
    # Debug folder for verification
    debug_dir = os.path.join(OUTPUT_DIR, 'debug_verification')
    os.makedirs(debug_dir, exist_ok=True)

    print("--- 2. Parsing JSON ---")
    polys_map = parse_polygons_from_json(JSON_FILE)
    all_images = list(polys_map.keys())
    print(f"Found {len(all_images)} images with labels in JSON.")

    if not all_images:
        print("ERROR: No images found in JSON. Check the file content.")
        return

    # Shuffle and Split (Prevent Data Leakage)
    random.shuffle(all_images)
    split_idx = int(len(all_images) * (1 - VAL_SPLIT))
    train_imgs = all_images[:split_idx]
    val_imgs = all_images[split_idx:]
    
    # To track stats
    stats = {'train_tiles': 0, 'val_tiles': 0, 'objects': 0}

    print("--- 3. Processing Images & Tiling ---")
    
    for split_name, img_list in [('train', train_imgs), ('val', val_imgs)]:
        for img_name in img_list:
            # Locate Image
            src_path = None
            # Check for common extensions
            for ext in ['.tif', '.tiff', '.jpg', '.png']:
                p = os.path.join(IMAGES_DIR, img_name + ext)
                if os.path.exists(p):
                    src_path = p
                    break
            
            if not src_path:
                print(f"[Warning] Image file not found for ID: {img_name}")
                continue

            # Read Huge Image
            huge_img = cv2.imread(src_path)
            if huge_img is None: 
                print(f"[Error] Could not read image: {src_path}")
                continue
            
            H, W = huge_img.shape[:2]
            img_polys = polys_map[img_name]

            # Sliding Window Loop
            for y in range(0, H, STRIDE):
                for x in range(0, W, STRIDE):
                    # Define Tile Box
                    x_end = min(x + TILE_SIZE, W)
                    y_end = min(y + TILE_SIZE, H)
                    
                    # Current Tile Geometry
                    cur_w = x_end - x
                    cur_h = y_end - y
                    
                    # Skip very small edge tiles (optional)
                    if cur_w < 50 or cur_h < 50:
                        continue

                    # The Tile Polygon in Global Coords
                    tile_poly = box(x, y, x_end, y_end)
                    
                    # Collect Labels for this Tile
                    tile_labels = []
                    
                    for poly in img_polys:
                        if not tile_poly.intersects(poly):
                            continue
                        
                        # Intersection (Clip)
                        inter = tile_poly.intersection(poly)
                        
                        # Filter tiny shards
                        if inter.area / poly.area < MIN_AREA_RATIO:
                            continue
                        
                        # Translate to Local Tile Coords (0,0)
                        # --- FIXED LINE BELOW ---
                        local_poly = translate(inter, xoff=-x, yoff=-y)
                        
                        # Safety check: is it empty?
                        if local_poly.is_empty: continue

                        # Convert to OBB String
                        label_str = get_yolo_obb_string(local_poly, cur_w, cur_h)
                        if label_str:
                            tile_labels.append(label_str)
                            stats['objects'] += 1

                    # Save Tile (Only if it has objects)
                    if len(tile_labels) > 0:
                        tile_filename = f"{img_name}_{x}_{y}"
                        
                        # Save Image
                        dst_img_path = os.path.join(OUTPUT_DIR, split_name, 'images', tile_filename + ".jpg")
                        tile_img = huge_img[y:y_end, x:x_end]
                        cv2.imwrite(dst_img_path, tile_img)
                        
                        # Save Label
                        dst_lbl_path = os.path.join(OUTPUT_DIR, split_name, 'labels', tile_filename + ".txt")
                        with open(dst_lbl_path, 'w') as f:
                            f.write("\n".join(tile_labels))
                            
                        stats[f'{split_name}_tiles'] += 1
                        
                        # Save a few verification images (first 20 objects total)
                        if stats['objects'] < 20: 
                            debug_path = os.path.join(debug_dir, f"DEBUG_{tile_filename}.jpg")
                            verify_and_draw(dst_img_path, dst_lbl_path, debug_path)

    print("\n" + "="*30)
    print("PROCESSING COMPLETE")
    print(f"Dataset generated at: {OUTPUT_DIR}")
    print(f"Train Tiles: {stats['train_tiles']}")
    print(f"Val Tiles:   {stats['val_tiles']}")
    print(f"Total Objects: {stats['objects']}")
    print(f"CHECK THE '{OUTPUT_DIR}/debug_verification' FOLDER NOW!")
    print("="*30)

if __name__ == "__main__":
    process_dataset()