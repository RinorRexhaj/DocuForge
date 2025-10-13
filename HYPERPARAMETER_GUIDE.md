# Quick Hyperparameter Tuning Guide

## 🎛️ Key Hyperparameters & Their Effects

### Model Architecture

| Parameter      | Default | Range   | Effect                                             |
| -------------- | ------- | ------- | -------------------------------------------------- |
| `dropout_rate` | 0.5     | 0.3-0.7 | Higher = less overfitting, but may reduce capacity |
| Input size     | 224x224 | 224-384 | Larger = more detail, slower training              |

### Training Configuration

| Parameter     | Default | Range        | Effect                                              |
| ------------- | ------- | ------------ | --------------------------------------------------- |
| `EPOCHS`      | 40      | 30-60        | More epochs = better convergence if not overfitting |
| `batch_size`  | 28      | 16-64        | Larger = faster, but needs more GPU memory          |
| Backbone LR   | 1e-4    | 5e-5 to 2e-4 | Lower = safer, higher = faster adaptation           |
| Attention LR  | 5e-4    | 2e-4 to 1e-3 | Controls attention learning speed                   |
| Classifier LR | 1e-3    | 5e-4 to 2e-3 | Higher = faster classifier training                 |

### Loss Function

| Parameter     | Default | Range   | Effect                                |
| ------------- | ------- | ------- | ------------------------------------- |
| Focal `alpha` | 1.5     | 0.5-2.0 | Higher = more weight on hard examples |
| Focal `gamma` | 2.5     | 1.0-5.0 | Higher = more focus on hard examples  |

### Scheduler

| Parameter              | Default | Range        | Effect                               |
| ---------------------- | ------- | ------------ | ------------------------------------ |
| `T_0` (restart period) | 5       | 3-10         | Smaller = more frequent LR resets    |
| `T_mult`               | 2       | 1-3          | Period multiplier after each restart |
| `eta_min`              | 1e-7    | 1e-8 to 1e-6 | Minimum learning rate                |

### Early Stopping

| Parameter   | Default | Range        | Effect                                 |
| ----------- | ------- | ------------ | -------------------------------------- |
| `patience`  | 15      | 10-25        | Higher = wait longer before stopping   |
| `min_delta` | 0.0003  | 0.0001-0.001 | Minimum improvement to count as better |

### Gradual Unfreezing

| Parameter       | Default  | Adjust To   | When                    |
| --------------- | -------- | ----------- | ----------------------- |
| Unfreeze layer2 | Epoch 8  | Epoch 5-10  | Earlier if underfitting |
| Unfreeze layer1 | Epoch 15 | Epoch 10-20 | Earlier if underfitting |
| Unfreeze all    | Epoch 25 | Epoch 20-30 | Earlier if underfitting |

## 🔧 Common Scenarios & Solutions

### Scenario 1: High Training Acc, Low Val Acc (Overfitting)

```python
# In model creation (Cell 7):
model = EnhancedResNetModel(num_classes=1, dropout_rate=0.6)  # Increase from 0.5

# In optimizer config (Cell 9):
optimizer = torch.optim.AdamW([
    {"params": backbone_params, "lr": 5e-5, "weight_decay": 2e-4},    # Reduce LR, increase decay
    {"params": attention_params, "lr": 2e-4, "weight_decay": 2e-4},
    {"params": classifier_params, "lr": 5e-4, "weight_decay": 2e-3}
], ...)
```

### Scenario 2: Both Training & Val Acc Low (Underfitting)

```python
# In training loop (Cell 10):
def unfreeze_layers(model, epoch):
    if epoch == 5:  # Unfreeze earlier (was 8)
        for name, param in model.named_parameters():
            if 'backbone.layer2' in name:
                param.requires_grad = True
    # ... rest remains same

# Increase learning rates:
optimizer = torch.optim.AdamW([
    {"params": backbone_params, "lr": 2e-4, "weight_decay": 5e-5},    # Increase
    {"params": attention_params, "lr": 8e-4, "weight_decay": 5e-5},
    {"params": classifier_params, "lr": 2e-3, "weight_decay": 5e-4}
], ...)

# Train longer:
EPOCHS = 50
```

### Scenario 3: Training Too Slow

```python
# Reduce batch size if GPU memory is full:
train_loader = DataLoader(train_dataset, batch_size=16, ...)  # Was 28

# Or simplify model slightly:
# In classifier, reduce layer sizes:
nn.Linear(in_features * 2, 512),  # Instead of 1024
nn.Linear(512, 256),
nn.Linear(256, 128),
nn.Linear(128, num_classes)
```

### Scenario 4: Stuck at ~85%, Can't Break 90%

```python
# Strategy 1: More aggressive augmentation
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(30),  # Increase from 20
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),  # More aggressive
    transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.5, hue=0.2),
    # ... rest remains same
])

# Strategy 2: Larger input size
IMG_SIZE = 256  # Instead of 224

# Strategy 3: More TTA augmentations
tta_acc, tta_f1, tta_auc, best_thresh = enhanced_test_time_augmentation(
    model, test_loader, device, num_tta=15  # Increase from 10
)

# Strategy 4: Focal loss adjustment for harder examples
criterion = FocalLoss(alpha=2.0, gamma=3.0, logits=True)  # More aggressive
```

### Scenario 5: Class Imbalance Issues

```python
# Check your class distribution first:
print(f"Class weights: {class_weights}")

# If heavily imbalanced (e.g., 80:20 split):
# Adjust focal loss alpha
criterion = FocalLoss(alpha=2.5, gamma=2.5, logits=True)  # Higher alpha

# Or use weighted sampler:
from torch.utils.data import WeightedRandomSampler

# Calculate sample weights
sample_weights = [class_weights[label] for _, label in train_dataset.samples]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

# Update dataloader:
train_loader = DataLoader(
    train_dataset,
    batch_size=28,
    sampler=sampler,  # Use sampler instead of shuffle
    pin_memory=True,
    num_workers=2
)
```

## 📊 Recommended Configurations by Dataset Size

### Small Dataset (< 1000 images)

```python
# More aggressive data augmentation
dropout_rate = 0.6
EPOCHS = 50
patience = 20

# Lower learning rates to prevent overfitting
backbone_lr = 5e-5
attention_lr = 2e-4
classifier_lr = 5e-4

# Freeze more layers initially
freeze_layers(model, freeze_until='layer4')  # Only train layer4 + heads
```

### Medium Dataset (1000-5000 images)

```python
# Current default configuration is optimal
dropout_rate = 0.5
EPOCHS = 40
patience = 15

backbone_lr = 1e-4
attention_lr = 5e-4
classifier_lr = 1e-3
```

### Large Dataset (> 5000 images)

```python
# Can afford more aggressive training
dropout_rate = 0.4
EPOCHS = 30
patience = 10

# Higher learning rates
backbone_lr = 2e-4
attention_lr = 8e-4
classifier_lr = 2e-3

# Unfreeze earlier
# Epoch 5: layer2
# Epoch 10: layer1
# Epoch 15: all
```

## 🎯 Progressive Fine-tuning Strategy

If you're not reaching 90%, try this step-by-step approach:

### Phase 1: Initial Training (Default Config)

```python
# Run with current settings
# Expected: 85-88% accuracy
```

### Phase 2: If Stuck, Increase Model Capacity

```python
# Unfreeze more layers earlier
# Increase training epochs to 50
# Adjust learning rates up by 50%
```

### Phase 3: Data-Centric Improvements

```python
# Review mislabeled samples in evaluation results
# Add more data augmentation
# Balance dataset if needed
```

### Phase 4: Advanced Techniques

```python
# Increase input size to 256 or 288
# Use more TTA augmentations (15-20)
# Ensemble multiple checkpoints (e.g., best 3 epochs)
```

## 💾 How to Apply Changes

### Method 1: Edit Cell Directly

1. Find the relevant cell (noted in each scenario)
2. Modify the parameter values
3. Re-run the cell and subsequent cells

### Method 2: Cell-by-Cell Adjustments

```python
# After loading model, before training:
# Adjust dropout
for module in model.modules():
    if isinstance(module, nn.Dropout):
        module.p = 0.6  # New dropout rate
```

## 🔍 Monitoring Training

Key metrics to watch:

- **Train/Val Accuracy Gap**: Should be < 10%
  - If > 15%: Overfitting → increase dropout, reduce LR
  - If < 5%: Good balance
- **Validation Loss**: Should decrease
  - If increasing: Overfitting or LR too high
  - If flat: Underfitting or LR too low
- **F1-Score**: Important for imbalanced data
  - If accuracy high but F1 low: Class imbalance issue
- **Learning Rate**: Watch the schedule
  - Should decrease gradually with restarts
  - If loss spikes: LR too high

## 🚀 Quick Start Commands

### Train with default settings:

Just run Cell 10

### Train with custom settings:

```python
# Modify these before Cell 10:
EPOCHS = 50
dropout_rate = 0.6

# Then run Cell 10
```

### Resume training from checkpoint:

```python
checkpoint = torch.load("saved_models/resnet_epoch_20.pth")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Continue training from epoch 21
```

---

**Remember**: The key to >90% accuracy is:

1. ✅ Clean, balanced data
2. ✅ Appropriate regularization
3. ✅ Sufficient training time
4. ✅ Test-Time Augmentation
5. ✅ Patience and iteration!
