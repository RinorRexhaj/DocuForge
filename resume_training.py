# Resume Training from Checkpoint
# Add this cell to your notebook to resume training from Epoch 26

import os
import torch

# Configuration
RESUME_FROM_EPOCH = 26  # Last good epoch before NaN
CHECKPOINT_PATH = f"saved_models/resnet_epoch_{RESUME_FROM_EPOCH}.pth"

# Check if checkpoint exists
if not os.path.exists(CHECKPOINT_PATH):
    print(f"❌ Checkpoint not found: {CHECKPOINT_PATH}")
    print(f"\nAvailable checkpoints:")
    for f in sorted(os.listdir("saved_models")):
        if f.startswith("resnet_epoch_") and f.endswith(".pth"):
            print(f"  - {f}")
else:
    print(f"✅ Found checkpoint: {CHECKPOINT_PATH}")
    
    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    
    # Restore model state
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Model state restored from epoch {checkpoint['epoch']+1}")
    
    # Restore optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"✅ Optimizer state restored")
    
    # Restore scheduler state
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"✅ Scheduler state restored")
    
    # Print checkpoint info
    print(f"\n📊 Checkpoint Information:")
    print(f"  Epoch: {checkpoint['epoch']+1}/{EPOCHS}")
    print(f"  Train Accuracy: {checkpoint.get('train_acc', 'N/A'):.4f}")
    print(f"  Train Loss: {checkpoint.get('train_loss', 'N/A'):.4f}")
    print(f"  Val Accuracy: {checkpoint.get('val_acc', 'N/A'):.4f}")
    print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    print(f"  Val F1: {checkpoint.get('val_f1', 'N/A'):.4f}")
    
    print(f"\n🚀 Ready to resume training from epoch {RESUME_FROM_EPOCH + 1}!")
    print(f"\n⚠️  IMPORTANT: Modify the training loop:")
    print(f"   Change: for epoch in range(EPOCHS):")
    print(f"   To:     for epoch in range({RESUME_FROM_EPOCH}, EPOCHS):")
