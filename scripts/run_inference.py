"""
Runs inference on 20 random images using the trained model and draws
bounding boxes with class labels and confidence scores.

Usage:
    1. First export the model (one-time):
       python models/models/research/object_detection/exporter_main_v2.py \
           --input_type=image_tensor \
           --pipeline_config_path=workspace/pipeline.config \
           --trained_checkpoint_dir=workspace/training_output \
           --output_directory=workspace/exported_model

    2. Then run this script:
       python scripts/run_inference.py
"""
import os
import glob
import random
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

# === CONFIGURATION ===
BASE_DIR = os.getcwd()
MODEL_DIR = os.path.join(BASE_DIR, 'workspace', 'exported_model', 'saved_model')
IMG_DIRS = [
    os.path.join(BASE_DIR, 'data', 'images', 'brugia'),
    os.path.join(BASE_DIR, 'data', 'images', 'wuchereria'),
]
OUTPUT_DIR = os.path.join(BASE_DIR, 'inference_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_IMAGES = 20
MIN_SCORE = 0.3  # Only draw boxes with confidence >= 30%

CLASS_NAMES = {1: 'Brugia malayi', 2: 'Wuchereria bancrofti'}
CLASS_COLORS = {1: (0, 255, 0), 2: (0, 165, 255)}  # Green, Orange

# === LOAD MODEL ===
print(f"Loading model from: {MODEL_DIR}")
if not os.path.exists(MODEL_DIR):
    print("\nERROR: Exported model not found!")
    print("Run this first:")
    print("  python models/models/research/object_detection/exporter_main_v2.py \\")
    print("      --input_type=image_tensor \\")
    print("      --pipeline_config_path=workspace/pipeline.config \\")
    print("      --trained_checkpoint_dir=workspace/training_output \\")
    print("      --output_directory=workspace/exported_model")
    exit(1)

model = tf.saved_model.load(MODEL_DIR)
detect_fn = model.signatures['serving_default']
print("Model loaded successfully!")

# === GATHER IMAGES ===
all_images = []
for img_dir in IMG_DIRS:
    if os.path.exists(img_dir):
        for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.png']:
            all_images.extend(glob.glob(os.path.join(img_dir, ext)))

if len(all_images) == 0:
    print("No images found!")
    exit(1)

n = min(NUM_IMAGES, len(all_images))
selected = random.sample(all_images, n)
print(f"\nRunning inference on {n} images...\n")

# === RUN INFERENCE ===
for i, img_path in enumerate(selected):
    # Load image
    img = Image.open(img_path).convert('RGB')
    img_np = np.array(img)
    input_tensor = tf.convert_to_tensor(img_np)[tf.newaxis, ...]

    # Detect
    detections = detect_fn(input_tensor)

    boxes = detections['detection_boxes'][0].numpy()       # [N, 4] in [ymin, xmin, ymax, xmax] normalized
    scores = detections['detection_scores'][0].numpy()     # [N]
    classes = detections['detection_classes'][0].numpy().astype(int)  # [N]

    # Draw results
    draw = ImageDraw.Draw(img)
    w, h = img.size
    det_count = 0

    for j in range(len(scores)):
        if scores[j] < MIN_SCORE:
            continue

        # Convert normalized coords to pixels
        ymin, xmin, ymax, xmax = boxes[j]
        x1 = int(xmin * w)
        y1 = int(ymin * h)
        x2 = int(xmax * w)
        y2 = int(ymax * h)

        cls_id = classes[j]
        cls_name = CLASS_NAMES.get(cls_id, f'Class {cls_id}')
        color = CLASS_COLORS.get(cls_id, (255, 0, 0))
        confidence = scores[j]

        # Draw box
        for thickness in range(5):
            draw.rectangle([x1-thickness, y1-thickness, x2+thickness, y2+thickness], outline=color)

        # Draw label background
        label = f"{cls_name}: {confidence:.0%}"
        label_bbox = draw.textbbox((x1, y1 - 25), label)
        draw.rectangle(label_bbox, fill=color)
        draw.text((x1, y1 - 25), label, fill=(0, 0, 0))

        det_count += 1

    # Save
    basename = os.path.basename(img_path)
    safe_name = os.path.splitext(basename)[0]
    out_path = os.path.join(OUTPUT_DIR, f"result_{i+1}_{safe_name}.jpg")
    img.save(out_path)
    print(f"[{i+1}/{n}] {basename}: {det_count} detections -> {out_path}")

print(f"\nDone! Check '{OUTPUT_DIR}/' for results.")
