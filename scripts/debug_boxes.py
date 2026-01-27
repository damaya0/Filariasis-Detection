import pandas as pd
import cv2
import os
import random

# --- CONFIGURATION ---
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data')
IMG_BASE_DIR = os.path.join(DATA_DIR, 'images')
OUTPUT_DIR = os.path.join(BASE_DIR, 'debug_output_v2')

# Create output folder
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Define datasets
DATASETS = [
    {
        "name": "Brugia",
        "csv": os.path.join(DATA_DIR, 'Brugia_Coordinates.csv'),
        "img_dir": os.path.join(IMG_BASE_DIR, 'brugia'), 
    },
    {
        "name": "Wuchereria",
        "csv": os.path.join(DATA_DIR, 'Wuchereria_Coordinates.csv'), 
        "img_dir": os.path.join(IMG_BASE_DIR, 'wuchereria'),
    }
]

def transform_coordinate(val, axis_size):
    """
    Transforms a 0-255 coordinate to actual image pixel space.
    Args:
        val: The coordinate from CSV (0-255)
        axis_size: The actual width or height of the image
        flip: Boolean, whether to flip the axis (for Y-axis)
    """
    # 1. Normalize (0 to 1)
    normalized = val / 255.0
    
    # 2. Scale to actual size
    scaled = normalized * axis_size
    

def save_debug_samples(dataset_info, num_samples=5, padding=15):
    if not os.path.exists(dataset_info['csv']):
        print(f"Error: CSV not found at {dataset_info['csv']}")
        return

    df = pd.read_csv(dataset_info['csv'])
    n = min(num_samples, len(df))
    sample_rows = df.sample(n=n)

    print(f"--- Processing {dataset_info['name']} ---")

    for index, row in sample_rows.iterrows():
        img_name = row['image_name']
        img_path = os.path.join(dataset_info['img_dir'], img_name)

        if not os.path.exists(img_path):
            continue

        image = cv2.imread(img_path)
        if image is None:
            continue
            
        # Get actual image dimensions
        h_actual, w_actual, _ = image.shape

        # --- TRANSFORM COORDINATES ---
        # X: Scale only
        x1 = transform_coordinate(row['end1_x'], w_actual)
        x2 = transform_coordinate(row['end2_x'], w_actual)
        
        # Y: Scale AND Flip (because CSV is bottom-left, Image is top-left)
        y1 = transform_coordinate(row['end1_y'], h_actual)
        y2 = transform_coordinate(row['end2_y'], h_actual)

        # --- CALCULATE BOX ---
        xmin_raw = min(x1, x2)
        xmax_raw = max(x1, x2)
        ymin_raw = min(y1, y2)
        ymax_raw = max(y1, y2)

        # Apply Padding
        xmin = max(0, xmin_raw - padding)
        ymin = max(0, ymin_raw - padding)
        xmax = min(w_actual, xmax_raw + padding)
        ymax = min(h_actual, ymax_raw + padding)

        # --- DRAW ---
        # Draw Endpoints (Yellow)
        cv2.circle(image, (x1, y1), 10, (0, 255, 255), -1)
        cv2.circle(image, (x2, y2), 10, (0, 255, 255), -1)

        # Draw Box (Green)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)

        # Save
        safe_name = img_name.replace('.jpg', '').replace('.JPG', '')
        out_filename = f"{dataset_info['name']}_{safe_name}_v2.jpg"
        cv2.imwrite(os.path.join(OUTPUT_DIR, out_filename), image)
        print(f"Saved: {out_filename}")

if __name__ == "__main__":
    save_debug_samples(DATASETS[0], num_samples=5)
    save_debug_samples(DATASETS[1], num_samples=5)
    print(f"\nDone! Check the '{OUTPUT_DIR}' folder.")