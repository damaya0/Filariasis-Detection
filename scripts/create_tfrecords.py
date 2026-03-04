import pandas as pd
import tensorflow as tf
import os
import io
from PIL import Image
from sklearn.model_selection import train_test_split
from object_detection.utils import dataset_util

# === CONFIGURATION ===
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data')
IMG_BASE_DIR = os.path.join(DATA_DIR, 'images')
WORKSPACE_DIR = os.path.join(BASE_DIR, 'workspace')

CSV_BRUGIA = os.path.join(DATA_DIR, 'rectangle_coordinates Brugia.csv')
CSV_WUCH = os.path.join(DATA_DIR, 'rectangle_coordinates WUCH.csv')

# Class names MUST match label_map.pbtxt exactly
CLASS_NAMES = {
    1: 'Brugia malayi',
    2: 'Wuchereria bancrofti'
}

DATASETS = [
    {
        "name": "Brugia",
        "csv": CSV_BRUGIA,
        "img_dir": os.path.join(IMG_BASE_DIR, 'brugia'),
        "id": 1
    },
    {
        "name": "Wuchereria",
        "csv": CSV_WUCH,
        "img_dir": os.path.join(IMG_BASE_DIR, 'wuchereria'),
        "id": 2
    }
]

def clean_filename(fname):
    fname = str(fname).strip()
    parts = fname.split('.')
    if len(parts) >= 3:
        return f"{parts[0]}.{parts[-1]}"
    return fname

def transform_x(val, width):
    return (val / 255.0) * width

def transform_y(val, height):
    return height - ((val / 255.0) * height)

def create_tf_example(base_filename, group_df, img_dir, class_id):
    img_path = os.path.join(img_dir, base_filename)

    # Read image
    with tf.io.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = base_filename.encode('utf8')
    image_format = b'jpg'
    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    classes_text, classes = [], []

    for index, row in group_df.iterrows():
        if 'Status' in row and row['Status'] != 'success':
            continue

        try:
            x1 = transform_x(row['BL_X'], width)
            x2 = transform_x(row['TR_X'], width)
            y1 = transform_y(row['BL_Y'], height)
            y2 = transform_y(row['TR_Y'], height)
        except (ValueError, KeyError):
            continue

        # Apply padding to capture worm thickness (endpoints define a thin line)
        PADDING = 50  # pixels in original image space
        xmin = max(0, min(x1, x2) - PADDING)
        xmax = min(width, max(x1, x2) + PADDING)
        ymin = max(0, min(y1, y2) - PADDING)
        ymax = min(height, max(y1, y2) + PADDING)

        # Normalize to [0, 1] for TFRecord
        xmins.append(xmin / width)
        xmaxs.append(xmax / width)
        ymins.append(ymin / height)
        ymaxs.append(ymax / height)
        classes_text.append(CLASS_NAMES[class_id].encode('utf8'))
        classes.append(class_id)

    # Skip images with no valid boxes
    if len(xmins) == 0:
        return None

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def process_and_save():
    all_examples = []

    if not os.path.exists(WORKSPACE_DIR):
        os.makedirs(WORKSPACE_DIR)

    for ds in DATASETS:
        if not os.path.exists(ds['csv']):
            print(f"Skipping {ds['name']} (CSV not found at {ds['csv']})")
            continue

        print(f"Processing {ds['name']} from CSV...")
        df = pd.read_csv(ds['csv'])

        # Clean filenames (strip .a/.b worm sub-identifiers)
        df['Base_Image'] = df['Filename'].apply(clean_filename)
        grouped = df.groupby('Base_Image')

        count = 0
        for base_img_name, group_df in grouped:
            img_path = os.path.join(ds['img_dir'], base_img_name)

            if not os.path.exists(img_path):
                continue

            tf_record = create_tf_example(base_img_name, group_df, ds['img_dir'], ds['id'])
            if tf_record is not None:
                all_examples.append(tf_record)
                count += 1

        print(f"  -> Added {count} images.")

    # Split Data (80% Train, 20% Test)
    train_data, test_data = train_test_split(all_examples, test_size=0.2, random_state=42)

    # Write TFRecords
    def write_record(data, name):
        path = os.path.join(WORKSPACE_DIR, name)
        writer = tf.io.TFRecordWriter(path)
        for example in data:
            writer.write(example.SerializeToString())
        writer.close()
        print(f"Saved {name}: {len(data)} examples")

    write_record(train_data, 'train.record')
    write_record(test_data, 'test.record')

    # Create Label Map
    label_map_content = """item {
      id: 1
      name: 'Brugia malayi'
    }
    item {
      id: 2
      name: 'Wuchereria bancrofti'
    }
    """
    with open(os.path.join(WORKSPACE_DIR, 'label_map.pbtxt'), 'w') as f:
        f.write(label_map_content)

if __name__ == "__main__":
    process_and_save()