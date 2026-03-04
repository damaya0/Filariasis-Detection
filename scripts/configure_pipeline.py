import tensorflow as tf
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2
import os
import urllib.request
import tarfile

BASE_DIR = os.getcwd()
WORKSPACE_DIR = os.path.join(BASE_DIR, 'workspace')
MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8'
PRE_TRAINED_DIR = os.path.join(WORKSPACE_DIR, 'pre_trained_model')

def setup_config():
    # 1. Download Model if not exists
    if not os.path.exists(PRE_TRAINED_DIR):
        print("Downloading base model...")
        os.makedirs(PRE_TRAINED_DIR, exist_ok=True)
        url = f'http://download.tensorflow.org/models/object_detection/tf2/20200711/{MODEL_NAME}.tar.gz'
        tar_path = os.path.join(PRE_TRAINED_DIR, 'model.tar.gz')
        urllib.request.urlretrieve(url, tar_path)
        
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=PRE_TRAINED_DIR)
        print("Download complete.")

    # 2. Read Base Config
    config_path = os.path.join(PRE_TRAINED_DIR, MODEL_NAME, 'pipeline.config')
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    
    with tf.io.gfile.GFile(config_path, "r") as f:
        text_format.Merge(f.read(), pipeline_config)

    # 3. Apply Customizations
    pipeline_config.model.ssd.num_classes = 2
    
    # BATCH SIZE: TF 2.13 CPU (Windows) is slow. Keep this low (2 or 4).
    pipeline_config.train_config.batch_size = 4
    pipeline_config.train_config.num_steps = 13000

    pipeline_config.train_input_reader.shuffle_buffer_size = 50
    
    # Paper Specifications (Focal Loss)
    pipeline_config.model.ssd.loss.classification_loss.weighted_sigmoid_focal.alpha = 0.25
    pipeline_config.model.ssd.loss.classification_loss.weighted_sigmoid_focal.gamma = 2.0

    # Override learning rate for batch_size=1
    lr = pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate
    lr.cosine_decay_learning_rate.learning_rate_base = 0.02
    lr.cosine_decay_learning_rate.warmup_learning_rate = 0.006666

    # Prevent random crop from cutting out the worm entirely
    for aug in pipeline_config.train_config.data_augmentation_options:
        if aug.HasField('random_crop_image'):
            aug.random_crop_image.min_object_covered = 0.5
    
    # Paths
    label_map_path = os.path.join(WORKSPACE_DIR, 'label_map.pbtxt')
    train_record_path = os.path.join(WORKSPACE_DIR, 'train.record')
    test_record_path = os.path.join(WORKSPACE_DIR, 'test.record')
    checkpoint_path = os.path.join(PRE_TRAINED_DIR, MODEL_NAME, 'checkpoint/ckpt-0')

    pipeline_config.train_input_reader.label_map_path = label_map_path
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [train_record_path]
    
    # EVALUATION SETTINGS (For Accuracy Printing)
    pipeline_config.eval_input_reader[0].label_map_path = label_map_path
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [test_record_path]
    
    pipeline_config.train_config.fine_tune_checkpoint = checkpoint_path
    pipeline_config.train_config.fine_tune_checkpoint_type = "detection"

    # Save
    out_config = os.path.join(WORKSPACE_DIR, 'pipeline.config')
    with tf.io.gfile.GFile(out_config, "wb") as f:
        f.write(text_format.MessageToString(pipeline_config))
    
    print(f"Configuration saved to: {out_config}")

if __name__ == "__main__":
    setup_config()