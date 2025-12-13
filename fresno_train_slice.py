import os
import cv2
import numpy as np
import shutil
import random
from shapely.geometry import Polygon, box

# ================= CONFIGURATION =================
# 1. Inputs
HUGE_IMAGES_DIR = './images'        # Your 5000x5000 images
HUGE_LABELS_DIR = './yolo_labels'   # Your generated .txt files

# 2. Outputs
OUTPUT_BASE = './yolo_dataset'      # Where the training data will go

# 3. Tiling Settings
TILE_SIZE = 640                     # Target size for YOLO (e.g., 640)
OVERLAP = 0.2                       # 20% overlap to catch edge objects
STRIDE = int(TILE_SIZE * (1 - OVERLAP))

# 4. Filtering
IOU_THRESHOLD = 0.3                 # Keep object if >30% is visible in tile
VAL_SPLIT = 0.2                     # 20% of data for validation
# =================================================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_yolo_obb_as_polygons(txt_path, img_w, img_h):
    """Reads YOLO OBB txt and converts to Shapely Polygons (Absolute Pixels)."""
    polys = []
    classes = []
    
    if not os.path.exists(txt_path):
        return [], []
        
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        coords = [float(x) for x in parts[1:]]
        
        # Denormalize
        pixel_pts = []
        for i in range(0, len(coords), 2):
            pixel_pts.append((coords[i] * img_w, coords[i+1] * img_h))
            
        if len(pixel_pts) >= 3:
            polys.append(Polygon(pixel_pts))
            classes.append(class_id)
            
    return polys, classes

def get_yolo_obb_str(shapely_poly, class_id, tile_w, tile_h):
    """Converts a Shapely Polygon back to normalized YOLO OBB string."""
    # 1. Get Minimum Rotated Rectangle of the clipped polygon
    rect = shapely_poly.minimum_rotated_rectangle
    
    # 2. Extract coords
    x, y = rect.exterior.coords.xy
    # shapely repeats the first point at the end, take first 4
    pts = list(zip(x, y))[:4]
    
    # 3. Normalize to Tile Size
    norm_vals = []
    for (px, py) in pts:
        nx = max(0.0, min(1.0, px / tile_w))
        ny = max(0.0, min(1.0, py / tile_h))
        norm_vals.extend([nx, ny])
        
    return f"{class_id} " + " ".join([f"{v:.6f}" for v in norm_vals])

def process_dataset():
    # Setup Output Directories
    for split in ['train', 'val']:
        ensure_dir(os.path.join(OUTPUT_BASE, split, 'images'))
        ensure_dir(os.path.join(OUTPUT_BASE, split, 'labels'))

    # Get List of Images
    image_files = [f for f in os.listdir(HUGE_IMAGES_DIR) if f.lower().endswith(('.jpg', '.png', '.tif', '.tiff'))]
    print(f"Found {len(image_files)} huge images.")

    for img_file in image_files:
        basename = os.path.splitext(img_file)[0]
        
        # Load Image
        img_path = os.path.join(HUGE_IMAGES_DIR, img_file)
        img = cv2.imread(img_path)
        if img is None: continue
        h_img, w_img = img.shape[:2]
        
        # Load Corresponding Labels
        label_path = os.path.join(HUGE_LABELS_DIR, basename + ".txt")
        polys, classes = load_yolo_obb_as_polygons(label_path, w_img, h_img)
        
        # Determine Split (Train or Val) based on Image
        # (We split by source image, NOT by tile, to prevent data leakage)
        split = 'val' if random.random() < VAL_SPLIT else 'train'
        
        # Sliding Window
        tile_count = 0
        
        for y in range(0, h_img, STRIDE):
            for x in range(0, w_img, STRIDE):
                # Calculate Crop Box
                x_end = min(x + TILE_SIZE, w_img)
                y_end = min(y + TILE_SIZE, h_img)
                cur_w = x_end - x
                cur_h = y_end - y
                
                # Create Tile Geometry
                tile_box = box(x, y, x_end, y_end)
                
                valid_labels = []
                
                # Check Intersections
                for poly, cls in zip(polys, classes):
                    if not tile_box.intersects(poly):
                        continue
                        
                    inter = tile_box.intersection(poly)
                    
                    # Area Threshold (Filter out tiny crumbs)
                    if inter.area / poly.area < IOU_THRESHOLD:
                        continue
                        
                    # Shift coordinates to Tile Frame (0,0 is top-left of tile)
                    # We translate geometry by (-x, -y)
                    local_poly = inter.translate(-x, -y)
                    
                    # Convert to OBB String
                    label_str = get_yolo_obb_str(local_poly, cls, cur_w, cur_h)
                    valid_labels.append(label_str)
                
                # Save Tile (Only if it has labels OR randomly keep some background)
                if len(valid_labels) > 0:
                    # Crop Image
                    tile_img = img[y:y_end, x:x_end]
                    
                    save_name = f"{basename}_{x}_{y}"
                    
                    # Save Image
                    cv2.imwrite(os.path.join(OUTPUT_BASE, split, 'images', save_name + ".jpg"), tile_img)
                    
                    # Save Label
                    with open(os.path.join(OUTPUT_BASE, split, 'labels', save_name + ".txt"), 'w') as f:
                        f.write("\n".join(valid_labels))
                        
                    tile_count += 1

        print(f"Processed {img_file} -> {split}: {tile_count} tiles created.")

    print("\nDataset preparation complete.")
    print(f"Location: {os.path.abspath(OUTPUT_BASE)}")

if __name__ == "__main__":
    process_dataset()