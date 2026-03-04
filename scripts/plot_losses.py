"""
Reads TensorBoard event files from training_output/ and plots loss curves.
Run after training completes:
    python scripts/plot_losses.py
"""
import os
import glob
import matplotlib.pyplot as plt

# Try importing TF's summary reader
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("Error: tensorboard not installed. Install with: pip install tensorboard")
    exit(1)

TRAIN_DIR = os.path.join(os.getcwd(), 'workspace', 'training_output', 'train')

# Find event files
event_files = glob.glob(os.path.join(TRAIN_DIR, 'events.out.tfevents.*'))
if not event_files:
    # Try parent directory
    TRAIN_DIR = os.path.join(os.getcwd(), 'workspace', 'training_output')
    event_files = glob.glob(os.path.join(TRAIN_DIR, 'events.out.tfevents.*'))

if not event_files:
    print(f"No event files found in workspace/training_output/")
    print("Make sure training has completed or is running.")
    exit(1)

print(f"Reading events from: {TRAIN_DIR}")

ea = EventAccumulator(TRAIN_DIR)
ea.Reload()

# Available scalar tags
available_tags = ea.Tags()['scalars']
print(f"Available tags: {available_tags}")

# Extract losses
loss_tags = {
    'Loss/classification_loss': ('Classification Loss', 'tab:blue'),
    'Loss/localization_loss': ('Localization Loss', 'tab:orange'),
    'Loss/regularization_loss': ('Regularization Loss', 'tab:green'),
    'Loss/total_loss': ('Total Loss', 'black'),
}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

for tag, (label, color) in loss_tags.items():
    if tag not in available_tags:
        print(f"  Warning: '{tag}' not found, skipping.")
        continue
    
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    
    # Top plot: raw losses
    ax1.plot(steps, values, label=label, color=color, alpha=0.7, linewidth=1)
    
    # Bottom plot: smoothed (moving average)
    window = max(1, len(values) // 20)
    if len(values) > window:
        smoothed = []
        for i in range(len(values)):
            start = max(0, i - window)
            smoothed.append(sum(values[start:i+1]) / (i - start + 1))
        ax2.plot(steps, smoothed, label=f'{label}', color=color, linewidth=2)

# Top plot formatting
ax1.set_ylabel('Loss Value')
ax1.set_title('Training Losses (Raw)')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Bottom plot formatting
ax2.set_xlabel('Training Steps')
ax2.set_ylabel('Loss Value')
ax2.set_title('Training Losses (Smoothed)')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# Also plot learning rate if available
if 'learning_rate' in available_tags:
    fig2, ax_lr = plt.subplots(figsize=(14, 4))
    events = ea.Scalars('learning_rate')
    steps = [e.step for e in events]
    values = [e.value for e in events]
    ax_lr.plot(steps, values, color='tab:red', linewidth=2)
    ax_lr.set_xlabel('Training Steps')
    ax_lr.set_ylabel('Learning Rate')
    ax_lr.set_title('Learning Rate Schedule')
    ax_lr.grid(True, alpha=0.3)
    plt.tight_layout()
    lr_path = os.path.join(os.getcwd(), 'learning_rate.png')
    fig2.savefig(lr_path, dpi=150)
    print(f"Saved: {lr_path}")

# Save loss plot
out_path = os.path.join(os.getcwd(), 'training_losses.png')
fig.savefig(out_path, dpi=150)
print(f"Saved: {out_path}")

plt.show()
