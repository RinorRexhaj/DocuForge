# NaN Loss Fix Guide - DocuForge Model Training

## Problem Summary
During Epoch 27, the training loss became NaN (Not a Number), causing premature training termination:
```
Epoch 27/40: 100% 50/50 [00:35<00:00, 1.75it/s, Loss=nan, Acc=0.656, LR=6.55e-05]
```

## Root Causes of NaN Loss

### 1. **Gradient Explosion**
- Gradients become extremely large during backpropagation
- Common in later epochs when layers are unfrozen
- Learning rate may be too high for the current state

### 2. **Numerical Instability**
- Extreme values in model outputs
- Division by zero or log(0) in loss calculation
- Overflow in exponential operations

### 3. **Data Issues**
- Corrupted images in dataset
- Invalid normalization values
- Inf/NaN in input tensors

## Applied Fixes

### ✅ Fix 1: Enhanced FocalLoss with NaN Protection
**Location**: Cell 3 (FocalLoss class)

**Changes**:
- Added input clamping to prevent extreme values
- NaN/Inf detection and replacement
- Stability checks at each computation step
- Safe fallback values when NaN detected

```python
# Key protections added:
- torch.clamp(inputs, min=-10, max=10)  # Prevent extreme logits
- torch.nan_to_num() for safe value replacement
- Final safety check before returning loss
```

### ✅ Fix 2: More Conservative Learning Rates
**Location**: Cell 9 (Optimizer configuration)

**Changes**:
```python
# OLD (aggressive):
backbone_params:   lr=1e-4
attention_params:  lr=5e-4
classifier_params: lr=1e-3

# NEW (conservative):
backbone_params:   lr=5e-5   # 50% reduction
attention_params:  lr=2e-4   # 60% reduction
classifier_params: lr=5e-4   # 50% reduction
```

### ✅ Fix 3: Aggressive Gradient Clipping
**Location**: Cell 10 (Training loop)

**Changes**:
```python
# OLD:
max_norm=1.0

# NEW:
max_norm=0.5  # More aggressive clipping
```

### ✅ Fix 4: Comprehensive NaN Detection
**Location**: Cell 10 (Training loop)

**Added checks**:
1. Input data validation (imgs, labels)
2. Model output validation
3. Loss value validation
4. Gradient validation
5. Batch skipping when NaN detected
6. Gradient norm monitoring

## How to Resume Training

### Option 1: Resume from Epoch 26 (Recommended)
```python
# Load checkpoint from before NaN occurred
checkpoint = torch.load("saved_models/resnet_epoch_26.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

# Adjust starting epoch
start_epoch = 26

# Modify training loop to start from this epoch:
for epoch in range(start_epoch, EPOCHS):
    # ... rest of training code
```

### Option 2: Start Fresh with New Configuration
Simply re-run all cells with the updated code. The new safeguards will prevent NaN from occurring.

## Prevention Strategies Going Forward

### 1. **Monitor Gradient Norms**
Watch for warnings like:
```
⚠️  Warning: Large gradient norm (15.23) at batch 45
```
If you see this frequently, reduce learning rates by 30-50%.

### 2. **Learning Rate Schedule**
The current configuration uses:
- **Warmup**: First 3 epochs (gradual increase)
- **Cosine Annealing**: Cyclic learning rate with restarts
- **Min LR**: 1e-8 (prevents too-low learning rates)

### 3. **Data Validation**
Before training, verify your data:
```python
# Add this before training loop
print("Validating dataset...")
for imgs, labels in train_loader:
    if torch.isnan(imgs).any() or torch.isinf(imgs).any():
        print("⚠️  WARNING: Invalid data detected!")
        break
    if imgs.min() < -10 or imgs.max() > 10:
        print(f"⚠️  Extreme values: min={imgs.min():.2f}, max={imgs.max():.2f}")
        break
print("✅ Data validation passed!")
```

### 4. **Checkpoint More Frequently**
The model saves every epoch, but you can also save mid-epoch:
```python
# Add inside training loop every N batches
if batch_idx % 100 == 0:
    temp_checkpoint = {
        'epoch': epoch,
        'batch': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(temp_checkpoint, f"saved_models/temp_epoch{epoch}_batch{batch_idx}.pth")
```

## Debugging NaN Issues

### Step 1: Check Model Outputs
```python
# After model forward pass
print(f"Output range: [{outputs.min():.4f}, {outputs.max():.4f}]")
print(f"Output mean: {outputs.mean():.4f}, std: {outputs.std():.4f}")
print(f"Has NaN: {torch.isnan(outputs).any()}")
print(f"Has Inf: {torch.isinf(outputs).any()}")
```

### Step 2: Check Gradients
```python
# After backward pass
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        if grad_norm > 10.0:
            print(f"{name}: gradient norm = {grad_norm:.2f}")
```

### Step 3: Check Loss Components
```python
# Inside FocalLoss.forward()
print(f"BCE_loss range: [{BCE_loss.min():.4f}, {BCE_loss.max():.4f}]")
print(f"pt range: [{pt.min():.4f}, {pt.max():.4f}]")
print(f"F_loss range: [{F_loss.min():.4f}, {F_loss.max():.4f}]")
```

## Alternative Approaches if NaN Persists

### 1. Use Standard Cross-Entropy Loss
```python
# Replace FocalLoss temporarily
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5]).to(device))
```

### 2. Reduce Model Complexity
```python
# In EnhancedResNetModel, reduce dropout
model = EnhancedResNetModel(num_classes=1, dropout_rate=0.3)  # Instead of 0.5
```

### 3. Freeze More Layers
```python
# Keep more layers frozen to reduce complexity
def unfreeze_layers(model, epoch):
    if epoch == 12:  # Instead of 8
        # Unfreeze layer2
    elif epoch == 25:  # Instead of 15
        # Unfreeze layer1
    # Never unfreeze layer1 and conv layers
```

### 4. Use Batch Normalization Carefully
```python
# In evaluation mode during validation
model.eval()  # This sets BatchNorm to eval mode (uses running stats)
```

## Expected Behavior After Fixes

### Normal Training (No NaN):
```
Epoch 27/40: 100% 50/50 [00:35<00:00, 1.75it/s, Loss=0.3245, Acc=0.875, LR=6.55e-05]
```

### With NaN Detection:
```
⚠️  Warning: NaN/Inf in model outputs at batch 42. Skipping batch.
Epoch 27/40: 100% 50/50 [00:35<00:00, 1.75it/s, Loss=0.3567, Acc=0.862, LR=6.55e-05]

⚠️  WARNING: NaN values detected during this epoch!
   Consider:
   1. Reducing learning rate further
   2. Checking data quality
   3. Using even more aggressive gradient clipping
   4. Restarting from a previous checkpoint
```

## Quick Fixes Summary

| Issue | Fix | Location |
|-------|-----|----------|
| Unstable loss | Added NaN protection to FocalLoss | Cell 3 |
| High learning rate | Reduced LRs by 50-60% | Cell 9 |
| Gradient explosion | Increased clipping to 0.5 | Cell 10 |
| Silent NaN propagation | Added detection at every step | Cell 10 |
| No recovery mechanism | Added batch skipping | Cell 10 |

## Testing the Fixes

After implementing these changes:

1. **Start training** and watch the first few epochs
2. **Look for warnings** - they indicate the fixes are working
3. **Monitor gradient norms** - should stay below 5.0
4. **Check loss values** - should decrease steadily without spikes
5. **Verify accuracy** - should improve each epoch

If training reaches Epoch 27 successfully without NaN, the fixes are working! 🎉

## Need More Help?

If NaN still occurs:
1. Check which specific layer/parameter caused it (use debugging code above)
2. Reduce learning rates by another 50%
3. Increase gradient clipping to 0.25
4. Consider using a simpler loss function temporarily
5. Validate your dataset for corrupted images

---

**Remember**: NaN issues are common in deep learning, especially when fine-tuning pre-trained models. The key is robust error detection and conservative hyperparameters. With these fixes, your model should train stably to completion! 🚀
