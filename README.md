# Filariasis Detection — Object Detection with TensorFlow

Detects **Brugia malayi** and **Wuchereria bancrofti** filarial worms in microscopy blood smear images using SSD MobileNetV2 FPN-Lite (TensorFlow Object Detection API).

---

## Prerequisites

- **Python 3.10** (tested with 3.10.19)
- **Windows 10/11** (tested on CPU; GPU optional)
- Microscopy images placed in `data/images/brugia/` and `data/images/wuchereria/`

---

## 1. Clone & Set Up Environment

```bash
git clone https://github.com/damaya0/Filariasis-Detection.git
cd Filariasis-Detection

# Create virtual environment
python -m venv tf_env
.\tf_env\Scripts\activate      # Windows
# source tf_env/bin/activate   # macOS/Linux

# Install dependencies
pip install tensorflow==2.10.1 tf-models-official==2.10.1
pip install pandas pillow matplotlib scikit-learn tensorboard
```

## 2. Install TF Object Detection API

```bash
# Clone the TF Models repo
git clone --depth 1 https://github.com/tensorflow/models.git models/models

# Install the Object Detection API
cd models/models/research
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
pip install .
cd ../../..
```

> **Note**: `protoc` (Protocol Buffers compiler) must be installed. Download from [GitHub](https://github.com/protocolbuffers/protobuf/releases) and add to PATH.

## 3. Prepare the Workspace

### 3.1 Add your images
Place microscopy images into these folders (not tracked by git):
```
data/images/brugia/        ← Brugia malayi images (.JPG)
data/images/wuchereria/    ← Wuchereria bancrofti images (.JPG)
```
The annotation CSVs (`data/rectangle_coordinates Brugia.csv` and `data/rectangle_coordinates WUCH.csv`) are already included in the repo.

### 3.2 Download pre-trained model
```bash
mkdir -p workspace/pre_trained_model
cd workspace/pre_trained_model

# Download SSD MobileNetV2 FPN-Lite 640x640 (COCO)
curl -L http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz -o model.tar.gz
tar -xzf model.tar.gz
cd ../..
```

### 3.3 Create label map
Create `workspace/label_map.pbtxt`:
```protobuf
item {
    id: 1
    name: 'Brugia malayi'
}
item {
    id: 2
    name: 'Wuchereria bancrofti'
}
```

### 3.4 Generate TFRecords
```bash
python scripts/create_tfrecords.py
```
This creates `workspace/train.record` (80% split) and `workspace/test.record` (20% split).

### 3.5 Generate pipeline config
```bash
python scripts/configure_pipeline.py
```
This generates `workspace/pipeline.config` with all paths and hyperparameters pre-configured:
- Batch size: 4
- Learning rate: 0.02 (cosine decay)
- Steps: 13,000

---

## 4. Train the Model

```bash
python models/models/research/object_detection/model_main_tf2.py \
    --model_dir=workspace/training_output \
    --pipeline_config_path=workspace/pipeline.config
```

- **CPU**: ~8 sec/step → 13,000 steps ≈ ~29 hours
- **GPU**: much faster (adjust batch_size up to 8-16 in `configure_pipeline.py`)
- Checkpoints are saved every 1,000 steps — you can safely stop (Ctrl+C) and resume later

### Monitor training
Look for decreasing `total_loss` in the console output:
```
INFO:tensorflow:{'Loss/classification_loss': 0.45,
 'Loss/localization_loss': 0.30, 'Loss/total_loss': 0.90}
```

---

## 5. Evaluate the Model

Run in a **separate terminal** (can run alongside training, or after):
```bash
python models/models/research/object_detection/model_main_tf2.py \
    --model_dir=workspace/training_output \
    --pipeline_config_path=workspace/pipeline.config \
    --checkpoint_dir=workspace/training_output
```

This prints COCO metrics (mAP@0.5, mAP@0.75, Recall, etc.) on the test set.

---

## 6. Run Inference (Visualize Detections)

### 6.1 Export the trained model
```bash
python models/models/research/object_detection/exporter_main_v2.py \
    --input_type=image_tensor \
    --pipeline_config_path=workspace/pipeline.config \
    --trained_checkpoint_dir=workspace/training_output \
    --output_directory=workspace/exported_model
```

### 6.2 Run inference on 20 random images
```bash
python scripts/run_inference.py
```

Results are saved to `inference_results/` with bounding boxes drawn:
- **Green** = Brugia malayi
- **Orange** = Wuchereria bancrofti

---

## 7. Plot Training Loss

After training completes (or is stopped), visualize the loss curves:
```bash
python scripts/plot_losses.py
```
Saves `training_losses.png` and `learning_rate.png` in the project root.

---

## Project Structure

```
Filariasis-Detection/
├── data/
│   ├── images/                          ← (gitignored) microscopy images
│   │   ├── brugia/
│   │   └── wuchereria/
│   ├── rectangle_coordinates Brugia.csv ← annotation coordinates
│   └── rectangle_coordinates WUCH.csv
├── scripts/
│   ├── configure_pipeline.py            ← generates pipeline.config
│   ├── create_tfrecords.py              ← converts images + CSVs → TFRecords
│   ├── plot_losses.py                   ← plots training loss curves
│   └── run_inference.py                 ← runs detection and draws boxes
├── workspace/                           ← (gitignored) training workspace
│   ├── label_map.pbtxt
│   ├── pipeline.config
│   ├── train.record / test.record
│   ├── pre_trained_model/
│   ├── training_output/
│   └── exported_model/
├── models/                              ← (gitignored) TF Models repo
└── .gitignore
```

## Tuning Tips

| Parameter | File | Recommended Range |
|---|---|---|
| `batch_size` | `configure_pipeline.py` | CPU: 1-4, GPU: 8-16 |
| `learning_rate_base` | `configure_pipeline.py` | 0.01-0.04 |
| `num_steps` | `configure_pipeline.py` | 5,000-50,000 |
| `MIN_SCORE` (inference threshold) | `run_inference.py` | 0.3-0.5 |
