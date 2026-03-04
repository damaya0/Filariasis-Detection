"""
Reads back a few examples from train.record and prints/visualizes the stored bounding boxes.
This verifies whether create_tfrecords.py wrote correct data.
"""
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
import io
import os

RECORD_PATH = os.path.join(os.getcwd(), 'workspace', 'train.record')
OUTPUT_DIR = os.path.join(os.getcwd(), 'debug_tfrecord')
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_SAMPLES = 5

dataset = tf.data.TFRecordDataset(RECORD_PATH)
feature_spec = {
    'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/width': tf.io.FixedLenFeature([], tf.int64),
    'image/filename': tf.io.FixedLenFeature([], tf.string),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/label': tf.io.VarLenFeature(tf.int64),
}

print(f"Reading from: {RECORD_PATH}\n")

for i, raw_record in enumerate(dataset.take(NUM_SAMPLES)):
    parsed = tf.io.parse_single_example(raw_record, feature_spec)

    filename = parsed['image/filename'].numpy().decode('utf8')
    height = parsed['image/height'].numpy()
    width = parsed['image/width'].numpy()

    xmins = tf.sparse.to_dense(parsed['image/object/bbox/xmin']).numpy()
    xmaxs = tf.sparse.to_dense(parsed['image/object/bbox/xmax']).numpy()
    ymins = tf.sparse.to_dense(parsed['image/object/bbox/ymin']).numpy()
    ymaxs = tf.sparse.to_dense(parsed['image/object/bbox/ymax']).numpy()
    labels = tf.sparse.to_dense(parsed['image/object/class/label']).numpy()

    print(f"=== Example {i+1}: {filename} ===")
    print(f"  Image size: {width}x{height}")
    print(f"  Num boxes: {len(xmins)}")

    if len(xmins) == 0:
        print("  WARNING: No bounding boxes!")
        continue

    # Decode image and draw boxes
    img_bytes = parsed['image/encoded'].numpy()
    img = Image.open(io.BytesIO(img_bytes))
    draw = ImageDraw.Draw(img)

    for j in range(len(xmins)):
        # Convert normalized [0,1] back to pixel coords
        x1 = int(xmins[j] * width)
        y1 = int(ymins[j] * height)
        x2 = int(xmaxs[j] * width)
        y2 = int(ymaxs[j] * height)
        box_w = x2 - x1
        box_h = y2 - y1

        print(f"  Box {j+1}: xmin={xmins[j]:.4f} ymin={ymins[j]:.4f} xmax={xmaxs[j]:.4f} ymax={ymaxs[j]:.4f}  "
              f"(pixels: [{x1},{y1}]->[{x2},{y2}], size={box_w}x{box_h})  label={labels[j]}")

        # Draw box on image
        draw.rectangle([x1, y1, x2, y2], outline='lime', width=5)
        draw.text((x1, y1 - 15), f"Class {labels[j]}", fill='lime')

    out_path = os.path.join(OUTPUT_DIR, f"tfrecord_sample_{i+1}_{filename}")
    img.save(out_path)
    print(f"  Saved: {out_path}\n")

print(f"Done! Check {OUTPUT_DIR}/ for visual output.")
