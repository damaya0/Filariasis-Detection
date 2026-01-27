import pandas as pd
import tensorflow as tf
import os
import io
from PIL import Image
from sklearn.model_selection import train_test_split
from object_detection.utils import dataset_util

# --- CONFIGURATION ---
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data')
IMG_BASE_DIR = os.path.join(DATA_DIR, 'images')
WORKSPACE_DIR = os.path.join(BASE_DIR, 'workspace')

# Define datasets with explicit paths
DATASETS = [
    {
        "name": "Brugia",
        "csv": os.path.join(DATA_DIR, 'Brugia_Coordinates.csv'),
        "img_dir": os.path.join(IMG_BASE_DIR, 'Brugia'), 
        "id": 1
    },
    {
        "name": "Wuchereria",
        "csv": os.path.join(DATA_DIR, 'Wuchereria_coordinates.csv'),
        "img_dir": os.path.join(IMG_BASE_DIR, 'Wuchereria'),
        "id": 2
    }
]

def create_tf_example(group, path, class_id):
    # Construct full image path
    img_path = os.path.join(path, group.filename)
    
    # 1. READ IMAGE
    with tf.io.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    classes_text, classes = [], []

    # 2. ITERATE CSV ROWS
    for index, row in group.object.iterrows():
        # Logic: Box is the rectangle defined by the two endpoints
        xmin = min(row['end1_x'], row['end2_x'])
        xmax = max(row['end1_x'], row['end2_x'])
        ymin = min(row['end1_y'], row['end2_y'])
        ymax = max(row['end1_y'], row['end2_y'])

        # Safety: Ensure box has width/height > 0
        if xmin == xmax: xmax += 1
        if ymin == ymax: ymax += 1

        # Normalize coordinates
        xmins.append(xmin / width)
        xmaxs.append(xmax / width)
        ymins.append(ymin / height)
        ymaxs.append(ymax / height)
        classes_text.append(str(class_id).encode('utf8'))
        classes.append(class_id)

    # 3. BUILD TF EXAMPLE
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
    
    # Ensure workspace exists
    if not os.path.exists(WORKSPACE_DIR): os.makedirs(WORKSPACE_DIR)

    for ds in DATASETS:
        if not os.path.exists(ds['csv']):
            print(f"Skipping {ds['name']} (CSV not found at {ds['csv']})")
            continue
            
        print(f"Processing {ds['name']} from CSV...")
        df = pd.read_csv(ds['csv'])
        
        # Group by image name 
        grouped = df.groupby('image_name')
        
        count = 0
        for filename, group in grouped:
            img_path = os.path.join(ds['img_dir'], filename)
            
            # Only process if image exists
            if os.path.exists(img_path):
                file_group = pd.Series({'filename': filename, 'object': group})
                tf_record = create_tf_example(file_group, ds['img_dir'], ds['id'])
                all_examples.append(tf_record)
                count += 1
            else:
                # Image in CSV but not in folder -> Skip
                pass
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