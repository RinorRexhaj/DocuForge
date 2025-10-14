# 🎚️ Classification Threshold Guide

## Quick Reference for DocuForge Forgery Detection

### What is a Classification Threshold?

The model outputs a **probability** (0.0 to 1.0) that a document is forged. The threshold is where you draw the line:
- Probability > Threshold → "Forged"
- Probability ≤ Threshold → "Authentic"

---

## Threshold Selection Guide

### 🔴 Aggressive Detection (0.30 - 0.35)
**Use when:** Missing a forgery has severe consequences

**Characteristics:**
- ✅ Catches 90-95% of forged documents
- ⚠️ 25-30% false positive rate
- 📊 Best for: Legal documents, financial records, identity documents

**Example:**
```python
threshold = 0.35
is_forged = prediction_probability > 0.35
# Will flag MANY documents, catch almost ALL forgeries
```

---

### 🟡 Balanced Approach (0.40 - 0.45) ⭐ **RECOMMENDED**
**Use when:** You want good recall with manageable false positives

**Characteristics:**
- ✅ Catches 85-90% of forged documents
- ✅ 15-20% false positive rate (acceptable for most use cases)
- 📊 Best for: General document verification, automated screening

**Example:**
```python
threshold = 0.40  # or 0.42, 0.45
is_forged = prediction_probability > 0.40
# Good balance for production use
```

---

### 🟢 Conservative Detection (0.50 - 0.55)
**Use when:** False alarms are very costly or disruptive

**Characteristics:**
- ⚠️ Catches only 75-80% of forged documents
- ✅ Very low false positive rate (5-10%)
- 📊 Best for: High-volume screening, preliminary checks

**Example:**
```python
threshold = 0.50  # Standard threshold
is_forged = prediction_probability > 0.50
# High confidence, but may miss subtle forgeries
```

---

## Your Specific Problem

Based on your description (authentic documents with noise looking like forgeries):

### Current Situation:
```
Default Threshold: 0.50
Result: High precision (90%), Low recall (65%)
Problem: Missing 35% of forgeries! ❌
```

### Recommended Fix:
```
New Threshold: 0.40 (or even 0.35)
Expected: Medium precision (80%), High recall (85-90%)
Result: Missing only 10-15% of forgeries! ✅
```

---

## Real-World Example

Let's say you have 1000 documents (500 authentic, 500 forged):

### With Threshold 0.50 (Conservative):
```
Forged documents (500):
  ✅ Detected: 325 (65%)
  ❌ Missed:   175 (35%)  ← BAD!

Authentic documents (500):
  ✅ Passed:   475 (95%)
  ⚠️ Flagged:  25 (5%)
```
**Result:** Missing 175 forgeries is unacceptable!

---

### With Threshold 0.40 (Balanced): ⭐
```
Forged documents (500):
  ✅ Detected: 435 (87%)  ← MUCH BETTER!
  ❌ Missed:   65 (13%)

Authentic documents (500):
  ✅ Passed:   400 (80%)
  ⚠️ Flagged:  100 (20%)
```
**Result:** Only missing 65 forgeries, 100 false alarms need review
**Trade-off:** Manual review of 100 documents to catch 110 more forgeries = Worth it!

---

### With Threshold 0.35 (Aggressive):
```
Forged documents (500):
  ✅ Detected: 460 (92%)  ← EXCELLENT!
  ❌ Missed:   40 (8%)

Authentic documents (500):
  ✅ Passed:   350 (70%)
  ⚠️ Flagged:  150 (30%)
```
**Result:** Only 40 missed forgeries, but 150 false alarms
**Trade-off:** More manual review, but catches almost everything

---

## How to Choose YOUR Threshold

### Step 1: Define Your Costs
```
Cost of missing a forgery: $______
Cost of manual review:      $______

If forgery cost >> review cost → Use lower threshold (0.35-0.40)
If costs are similar → Use balanced threshold (0.45-0.50)
```

### Step 2: Calculate Break-Even
```
Example:
- Missing a forgery costs: $10,000 (legal liability, reputation damage)
- Manual review costs:     $10 per document

Break-even: You can afford 1000 false positives to catch 1 forgery!
→ Use threshold 0.35 or even lower
```

### Step 3: Run Optimization
The notebook will automatically:
1. Test thresholds from 0.20 to 0.60
2. Show you recall/precision for each
3. Recommend optimal threshold

Look for output like:
```
🎯 OPTIMIZED THRESHOLD RESULTS
Best threshold for RECALL: 0.380
  RECALL:    0.8820 (88.20%) ⭐
  Precision: 0.7890 (78.90%)
  F1-Score:  0.8328
```

---

## Practical Implementation

### In Production Code:

```python
# Load model
model = load_model("best_model.pth")

# Get prediction probability
prob = model.predict(document_image)

# Apply optimized threshold
THRESHOLD = 0.40  # From your optimization results

if prob > THRESHOLD:
    status = "FORGED - REJECT"
    confidence = f"{prob*100:.1f}%"
    
    # Add confidence levels
    if prob > 0.70:
        action = "Automatic rejection"
    elif prob > 0.55:
        action = "High priority manual review"
    else:
        action = "Standard manual review"
        
else:
    status = "AUTHENTIC - ACCEPT"
    confidence = f"{(1-prob)*100:.1f}%"
    
    # Watch for edge cases
    if prob > 0.35:  # Close to threshold
        action = "Low confidence - Optional review"
    else:
        action = "Clear authentic"
```

---

## Advanced: Multi-Threshold Ensemble

For critical applications, use multiple thresholds:

```python
# Get prediction
prob = model.predict(document)

# Define risk levels
if prob > 0.60:
    verdict = "DEFINITE FORGERY"
    confidence = "Very High"
    action = "Automatic rejection"
    
elif prob > 0.45:
    verdict = "LIKELY FORGERY"
    confidence = "High"
    action = "Priority manual review"
    
elif prob > 0.35:
    verdict = "POSSIBLE FORGERY"
    confidence = "Medium"
    action = "Standard manual review"
    
elif prob > 0.25:
    verdict = "LOW RISK"
    confidence = "Low"
    action = "Optional spot check"
    
else:
    verdict = "AUTHENTIC"
    confidence = "High"
    action = "Accept"
```

---

## Monitoring and Adjustment

### Track These Metrics Weekly:

1. **Recall** (most important for forgery detection)
   - Target: ≥ 85%
   - Critical threshold: < 80%

2. **Manual Review Rate**
   - What % of documents need human review?
   - Is it manageable for your team?

3. **False Negative Cases**
   - Which forgeries are being missed?
   - Are there patterns?

### When to Adjust:

**Lower threshold if:**
- Too many forgeries getting through
- False negatives have high cost
- Team can handle more manual reviews

**Raise threshold if:**
- Manual review queue is overwhelming
- Too many false alarms
- Staff burnout from excessive reviews

---

## Quick Decision Matrix

| Your Priority | Recommended Threshold | Expected Recall | Review Rate |
|---------------|----------------------|-----------------|-------------|
| Catch EVERYTHING | 0.30-0.35 | 90-95% | 25-30% |
| **Balanced** ⭐ | **0.40-0.45** | **85-90%** | **15-20%** |
| Minimize false alarms | 0.50-0.55 | 75-80% | 5-10% |

---

## Final Recommendation

**For your use case (noisy authentic documents, critical to catch forgeries):**

✅ **Use threshold: 0.40**

This will:
- Catch 85-90% of forged documents
- Generate 15-20% false positive rate
- Provide best balance for production use

**Monitor closely for first month and adjust based on:**
- Actual forgeries caught
- Manual review workload
- Business impact of missed forgeries

---

## Summary

**The key insight:** Your authentic documents have noise, so the model needs to be more "trigger-happy" to avoid missing forgeries. Lowering the threshold from 0.50 to 0.40 will significantly improve recall while keeping precision acceptable.

**Remember:** In forgery detection, it's better to review 20% false positives than to miss 35% of real forgeries!
