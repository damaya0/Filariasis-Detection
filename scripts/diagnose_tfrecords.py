"""
Deep diagnostic: reads ALL examples from train.record and reports statistics
to help debug localization_loss = 0.0
"""
import tensorflow as tf
import numpy as np
import os

RECORD_PATH = os.path.join(os.getcwd(), 'workspace', 'train.record')

feature_spec = {
    'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/width': tf.io.FixedLenFeature([], tf.int64),
    'image/filename': tf.io.FixedLenFeature([], tf.string),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    'image/object/class/text': tf.io.VarLenFeature(tf.string),
}

dataset = tf.data.TFRecordDataset(RECORD_PATH)

total_images = 0
total_boxes = 0
images_with_zero_boxes = 0
degenerate_boxes = 0  # zero or negative area
box_widths = []
box_heights = []
box_areas = []
labels_seen = set()
class_text_seen = set()

# Resized dimensions (what the model sees)
RESIZE_W = 640
RESIZE_H = 640

print(f"Scanning: {RECORD_PATH}\n")

for raw_record in dataset:
    parsed = tf.io.parse_single_example(raw_record, feature_spec)

    height = parsed['image/height'].numpy()
    width = parsed['image/width'].numpy()
    filename = parsed['image/filename'].numpy().decode('utf8')

    xmins = tf.sparse.to_dense(parsed['image/object/bbox/xmin']).numpy()
    xmaxs = tf.sparse.to_dense(parsed['image/object/bbox/xmax']).numpy()
    ymins = tf.sparse.to_dense(parsed['image/object/bbox/ymin']).numpy()
    ymaxs = tf.sparse.to_dense(parsed['image/object/bbox/ymax']).numpy()
    labels = tf.sparse.to_dense(parsed['image/object/class/label']).numpy()
    texts = tf.sparse.to_dense(parsed['image/object/class/text'], default_value=b'').numpy()

    total_images += 1

    if len(xmins) == 0:
        images_with_zero_boxes += 1
        continue

    for i in range(len(xmins)):
        labels_seen.add(int(labels[i]))
        if i < len(texts):
            class_text_seen.add(texts[i].decode('utf8'))

        # Box dimensions in resized 640x640 space
        bw = (xmaxs[i] - xmins[i]) * RESIZE_W
        bh = (ymaxs[i] - ymins[i]) * RESIZE_H
        area = bw * bh

        if bw <= 0 or bh <= 0:
            degenerate_boxes += 1

        box_widths.append(bw)
        box_heights.append(bh)
        box_areas.append(area)
        total_boxes += 1

        # Flag suspicious boxes
        if xmins[i] < 0 or ymins[i] < 0 or xmaxs[i] > 1.0 or ymaxs[i] > 1.0:
            print(f"  OUT OF RANGE: {filename} box {i}: [{xmins[i]:.4f},{ymins[i]:.4f},{xmaxs[i]:.4f},{ymaxs[i]:.4f}]")
        if xmins[i] >= xmaxs[i] or ymins[i] >= ymaxs[i]:
            print(f"  INVERTED: {filename} box {i}: [{xmins[i]:.4f},{ymins[i]:.4f},{xmaxs[i]:.4f},{ymaxs[i]:.4f}]")

box_widths = np.array(box_widths)
box_heights = np.array(box_heights)
box_areas = np.array(box_areas)

print("=" * 60)
print("DATASET SUMMARY")
print("=" * 60)
print(f"Total images:              {total_images}")
print(f"Images with ZERO boxes:    {images_with_zero_boxes}")
print(f"Total boxes:               {total_boxes}")
print(f"Degenerate boxes (<=0 area): {degenerate_boxes}")
print(f"Labels seen:               {labels_seen}")
print(f"Class text values seen:    {class_text_seen}")

if total_boxes > 0:
    print(f"\n--- Box sizes in 640x640 resized space ---")
    print(f"Width:  min={box_widths.min():.1f}  max={box_widths.max():.1f}  mean={box_widths.mean():.1f}  median={np.median(box_widths):.1f}")
    print(f"Height: min={box_heights.min():.1f}  max={box_heights.max():.1f}  mean={box_heights.mean():.1f}  median={np.median(box_heights):.1f}")
    print(f"Area:   min={box_areas.min():.0f}  max={box_areas.max():.0f}  mean={box_areas.mean():.0f}")

    print(f"\n--- Box size distribution (640x640 space) ---")
    tiny = np.sum(box_areas < 32*32)
    small = np.sum((box_areas >= 32*32) & (box_areas < 96*96))
    medium = np.sum((box_areas >= 96*96) & (box_areas < 256*256))
    large = np.sum(box_areas >= 256*256)
    print(f"Tiny   (<32x32 area):    {tiny}")
    print(f"Small  (32x32 - 96x96):  {small}")
    print(f"Medium (96x96 - 256x256): {medium}")
    print(f"Large  (>256x256):       {large}")

    # Check boxes per image distribution
    print(f"\n--- Boxes per image ---")
    print(f"Average: {total_boxes / total_images:.1f}")
