"""
ONE-CLICK SETUP & TRAIN SCRIPT
Run from the project root:
    python scripts/setup_and_train.py
"""
import subprocess
import os
import sys

BASE_DIR = os.getcwd()
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'models')

def run(cmd, desc):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, cwd=BASE_DIR)
    if result.returncode != 0:
        print(f"FAILED: {desc}")
        sys.exit(1)

# Step 1: Clone TF Models repo if not present
if not os.path.exists(MODELS_DIR):
    print("Cloning TensorFlow Models repository...")
    os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
    run('git clone --depth 1 https://github.com/tensorflow/models.git models/models',
        'Cloning TF Models')
    
    # Install Object Detection API
    run(f'cd models/models/research && protoc object_detection/protos/*.proto --python_out=. && pip install .',
        'Installing Object Detection API')
else:
    print("TF Models repo already exists, skipping clone.")

# Step 2: Verify TF + Object Detection API
run(f'{sys.executable} scripts/test.py', 'Verifying TensorFlow installation')

# Step 3: Generate TFRecords
run(f'{sys.executable} scripts/create_tfrecords.py', 'Creating TFRecords')

# Step 4: Configure pipeline
run(f'{sys.executable} scripts/configure_pipeline.py', 'Configuring training pipeline')

# Step 5: Train
print(f"\n{'='*60}")
print("  STARTING TRAINING")
print(f"{'='*60}")
print("Training will run for 13,000 steps.")
print("Monitor with TensorBoard: tensorboard --logdir=workspace/training_output\n")

train_script = os.path.join(MODELS_DIR, 'research', 'object_detection', 'model_main_tf2.py')
run(f'{sys.executable} {train_script} '
    f'--model_dir=workspace/training_output '
    f'--pipeline_config_path=workspace/pipeline.config',
    'Training')

print("\n✅ Training complete!")
print("Run 'python scripts/plot_losses.py' to see loss curves.")
