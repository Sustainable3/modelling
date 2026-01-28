'''
Docstring for tree_area.trees_inference

Segformer segmentation inference with results transformation
in a GPU-efficient scheme

EP, genAI adapted by MD
I 26
'''

import numpy as np
import torch
import os
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
import pandas as pd
import argparse
import math
from time import time

DATA = '/users/project1/pt01299/dane_syntetyczne/images'
BS = 256

def calculate_tree_area(input_path, device, batch_size=BS):
    t = time()
    results = []
    
    # Model configuration
    print("Loading model...")
    processor = AutoImageProcessor.from_pretrained("restor/tcd-segformer-mit-b2", use_fast=True)
    model = SegformerForSemanticSegmentation.from_pretrained("restor/tcd-segformer-mit-b2").to(device)
    model.eval()

    image_files = [
        f for f in os.listdir(input_path)
        if f.lower().endswith(".jpg")
    ]

    if not image_files:
        print("No JPG images found in:", input_path)
        return

    total_images = len(image_files)
    total_batches = math.ceil(total_images / batch_size)
    print(f"Found {total_images} images. Processing in {total_batches} batches (Batch size: {batch_size})...")

    # Iterate through files in chunks (batches)
    for i in range(0, total_images, batch_size):
        current_batch_num = (i // batch_size) + 1
        print(f"Processing batch {current_batch_num}/{total_batches}...", end="\r")

        batch_filenames = image_files[i : i + batch_size]
        batch_images = []
        original_sizes = []

        # Load images for the current batch
        for img_name in batch_filenames:
            img_path = os.path.join(input_path, img_name)
            image = Image.open(img_path).convert("RGB")
            batch_images.append(image)
            original_sizes.append(image.size) 

        # Preprocess the batch
        # processor accepts a list of images and returns a batch tensor
        inputs = processor(images=batch_images, return_tensors="pt", do_resize=True, size=(320, 320))
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Batch Prediction
        with torch.no_grad():
            outputs_sf = model(**inputs)

        # Retrieve masks for the batch
        batch_masks = outputs_sf.logits.argmax(dim=1).cpu().numpy()

        # Process each result in the batch separately
        for idx, mask in enumerate(batch_masks):
            img_name = batch_filenames[idx]
            original_size = original_sizes[idx]
            
            # Resize mask back to ORIGINAL image size
            mask_img = Image.fromarray(mask.astype(np.uint8))
            mask_resized = np.array(mask_img.resize(original_size, resample=Image.NEAREST))

            total_pixels = mask_resized.size
            tree_pixels = np.sum(mask_resized == 1)

            if total_pixels > 0:
                tree_area_share = tree_pixels / total_pixels
            else:
                tree_area_share = 0

            results.append({
                "image": img_name,
                "tree_area_share": tree_area_share,
                "tree_area_percent": tree_area_share * 100
            })

    print("\nProcessing2 complete.", time()-t)
    
    # Save results
    df = pd.DataFrame(results)
    output_filename = "test_tree2a.csv"
    df.to_csv(output_filename, index=False)
    print(f"Results saved to {output_filename}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate tree area share using SegFormer"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=DATA,
        help="Path to directory with JPG images"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BS,
        help="Number of images to process at once (default: 8)"
    )
    return parser.parse_args()


def parse_args1():
    parser = argparse.ArgumentParser(
        description="Calculate tree area share using SegFormer"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="DATA",
        help="Path to directory with JPG images"
    )
    return parser.parse_args()


def calculate_tree_area1(input_path, device):
    t = time()
    results = []
    # Model konfiguracja
    processor = AutoImageProcessor.from_pretrained("restor/tcd-segformer-mit-b2", use_fast=True)
    model = SegformerForSemanticSegmentation.from_pretrained("restor/tcd-segformer-mit-b2").to(device)
    model.eval()

    image_files = [
        f for f in os.listdir(input_path)
        if f.lower().endswith(".jpg")
    ]

    if not image_files:
        print("No JPG images found in:", input_path)
        return

    for img_name in image_files:
        img_path = os.path.join(input_path, img_name)
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt", do_resize=True, size=(320, 320))
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Predykcja i stworzenie maski
        with torch.no_grad():
            outputs_sf = model(**inputs)

        mask = outputs_sf.logits.argmax(dim=1).squeeze().cpu().numpy()
        mask_resized = np.array(Image.fromarray(mask.astype(np.uint8)).resize(image.size))

        total_pixels = mask_resized.size
        tree_pixels = np.sum(mask_resized == 1)

        # Obliczenie udziaĹ‚u powierzchni
        tree_area_share = tree_pixels / total_pixels

        results.append({
            "image": img_name,
            "tree_area_share": tree_area_share,
            "tree_area_percent": tree_area_share * 100
        })
    df = pd.DataFrame(results)
    print(df)
    df.to_csv("test_tree1a.csv", index=False)
    print("\nProcessing1 complete.", time()-t)



if __name__ == "__main__":
    args = parse_args()
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU.")

    calculate_tree_area(args.input_dir, device, args.batch_size)
    
    # args = parse_args1()
    # device = torch.device("cuda")
    # print(f"Using device: {device}")

    # calculate_tree_area1(args.input_dir, device)