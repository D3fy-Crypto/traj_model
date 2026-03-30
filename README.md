# GRU Trajectory Similarity Learning

## Overview

This implementation provides a **minimal PyTorch training pipeline** for trajectory representation learning using:
- **GRU encoder** for encoding variable-length trajectory prefixes and future segments
- **Weighted multi-prefix InfoNCE loss** for contrastive similarity learning
- **Timestep-based weighting** to handle the ambiguity of short prefixes

## Problem Statement

### What Problem Are We Solving?

Given a large corpus of navigation trajectories, we want to learn an **embedding space** where:
- A trajectory **prefix** is **close** to its true **future segment**
- A trajectory **prefix** is **far** from future segments of **other trajectories**

This is **not a classification task** (assigning a trajectory to a discrete class), but rather a **similarity learning / retrieval** task (learning to match prefixes to their continuations).

### Why Similarity Learning Instead of Classification?

Classification requires fixed semantic classes—we'd have to treat each trajectory as a class label. This is problematic because:

1. **Test-time generalization**: At test time, we may encounter entirely new trajectories not seen during training. Classification cannot handle unseen classes.
2. **Task mismatch**: The real task is to *match* or *retrieve* continuation segments, not assign class labels.
3. **Scalability**: Creating a new class for every trajectory doesn't scale to large datasets with thousands of unique trajectories.

Similarity learning (contrastive learning) is appropriate because:
- It learns a representation space based on **similarity**, not class identity
- New, unseen trajectories can be handled naturally by embedding them
- It directly optimizes the task we care about: matching prefixes to continuations

---

## Why InfoNCE Instead of Plain Cross-Entropy?

Both approaches can be viewed as contrastive losses, but they differ:

### Plain Cross-Entropy over Trajectory IDs
```
CE(logits, target_traj_id)
```
This treats each trajectory as a fixed semantic class and tries to classify "which trajectory does this prefix come from?" This is **not what we want** because:
- It conditions the model on trajectory identity, not trajectory content
- It assumes each trajectory is a distinct class, which is problematic for test-time trajectories

### InfoNCE Loss
```
logits = z_prefix @ z_future.T / temperature
CE(logits, target_future_index)
```
This is **similarity-based contrastive learning**. It directly optimizes:
- **Positive pairs**: prefix should have high similarity to its true future
- **Negative pairs**: prefix should have low similarity to futures from other trajectories

The key insight: **we're not classifying trajectory ID; we're learning to match content**.

---

## Why Use Many Prefixes Instead of One Fixed Prefix?

### Training on a Single Fixed Prefix
If we only trained on, e.g., "first 5 steps → next 5 steps":
- The model sees supervision only for this **one specific prefix length**
- At test time, if queried from a different prefix length (e.g., 10 steps, or 2 steps), the model may perform poorly
- We waste information in the trajectory by not using all available prefix points

### Training on Many Prefixes
We sample multiple prefix endpoints across each trajectory:
- **Prefix ending at t=2**: `x[0:2] → x[2:7]`
- **Prefix ending at t=5**: `x[0:5] → x[5:10]`
- **Prefix ending at t=8**: `x[0:8] → x[8:13]`
- ... and so on

This teaches the model to work from **any prefix length**, matching real-world deployment where the model can be queried from variable amounts of history.

---

## Why Keep Early Prefixes but Weight Them Less?

### The Problem with Very Short Prefixes
Very early in a trajectory, multiple futures may still be plausible. For example:
- After 2 steps, the agent might go left, right, or straight
- After 10 steps, the trajectory is more constrained and the future is more predictable

### Solution: Timestep-Based Weighting
We keep both early and late prefixes in the training data, but use **loss weights** to control their contribution:

For a trajectory of length $L_i$ and prefix endpoint $t$:
- **Progress**: $\text{progress} = \frac{t - t_{\min}}{L_i - t_{\min} - f_{\text{len}}}$ (normalized to $[0, 1]$)
- **Weight**: $w(t) = w_{\min} + (1 - w_{\min}) \cdot \text{progress}$

With $w_{\min} = 0.3$:
- Early prefixes get weight $\approx 0.3$ (still learn from them, but less)
- Late prefixes get weight $\approx 1.0$ (more informative, higher weight)

This improves training by:
1. **Using all available data** (don't throw away early timesteps)
2. **Avoiding dominance by ambiguous pairs** (don't overweight early, ambiguous prefixes)
3. **Focusing on informative pairs** (late prefixes have more signal)

---

## Negatives from Other Trajectories

We use **negatives from other trajectories in the batch**, not from the same trajectory.

### Why Not Same-Trajectory Negatives?
Within one trajectory, multiple futures might still be semantically related:
- Different future segments from the same navigation path share similar terrain, direction, etc.
- Using them as negatives could confuse contrastive learning (a prefix might be legitimately close to a non-target future from the same trajectory)

### Why Batch-Level Negatives?
- **Each prefix in the batch**: treated as a positive for its own future
- **All other futures in the batch**: treated as negatives (they come from different trajectories)
- **Cross-batch negatives could be added** but batch-level is sufficient and simpler

This design ensures the model learns that **different trajectories have different futures**, while **the same trajectory's prefix matches its continuation**.

---

## Architecture

### GRU Encoder

The `GRUEncoder` processes variable-length trajectory sequences:

```python
Input:  sequences [B, T_max, D], lengths [B]
  ↓
pack_padded_sequence (ignore padding during RNN)
  ↓
GRU (hidden_dim=64)
  ↓
Extract final hidden state [B, 64]
  ↓
Projection head: Linear(64→64) → ReLU → Linear(64→32)
  ↓
L2-normalize → embeddings [B, 32]
```

**Key design decisions**:
- **Variable-length handling**: `pack_padded_sequence` only processes valid timesteps
- **Final state only**: We use the GRU's final hidden state (not attention over all states)
- **Small projection head**: Keeps parameters low, improves generalization
- **L2-normalization**: Ensures embeddings lie on a unit sphere (standard for similarity learning)

**Why this architecture for trajectory data?**
- GRU is compact and effective for sequential data
- Final hidden state captures the full trajectory context (order matters)
- Normalization makes similarity scores comparable across batches

---

### Training Loop

1. **Sample multi-prefix pairs**: For each trajectory, sample multiple (prefix, future) pairs with variable prefix lengths.
2. **Encode**: Run GRU encoder on prefixes and futures to get normalized embeddings.
3. **Compute similarity matrix**: Dot product between all prefixes and all futures.
4. **InfoNCE loss**: Cross-entropy over the similarity matrix, with diagonal as positives.
5. **Weight by timestep**: Multiply loss by progress-based weights (early → lower, late → higher).
6. **Backward pass**: Standard gradient descent with Adam optimizer.
7. **Metrics**: Compute top-1 and top-5 retrieval accuracy (how often is the true future in the top-k matches?).

### Scheduler

We use `CosineAnnealingLR` for a **global learning rate schedule** over the full training. This is a standard choice for contrastive learning and smoothly anneals the learning rate.

---

## Retrieval Metrics

**Top-1 and Top-5 Accuracy**: For each prefix embedding, we rank all future embeddings by similarity. We then check:
- **Top-1**: Is the true future ranked 1st?
- **Top-5**: Is the true future in the top 5?

These measure how well the learned embeddings capture trajectory continuation.

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_dim` | 3 | Trajectory feature dimension: [dx, dyaw·sin(θ), dyaw·cos(θ)] |
| `hidden_dim` | 64 | GRU hidden state size |
| `embedding_dim` | 32 | Final embedding size |
| `min_prefix_len` | 2 | Minimum prefix length to sample |
| `future_len` | 5 | Target future segment length |
| `max_prefixes_per_traj` | 5 | Max prefix pairs per trajectory (None = use all) |
| `w_min` | 0.3 | Minimum weight for early prefixes |
| `temperature` | 0.1 | Softmax temperature for similarity scaling |
| `lr` | 1e-3 | Learning rate (Adam) |
| `T_max` | 10 | Max steps for cosine annealing scheduler |

---

## How It All Fits Together

```
Raw Trajectories (motion sequences)
    ↓
Feature Encoding (dx, dyaw·sin, dyaw·cos)
    ↓
Padding to batch size
    ↓
Sample Multiple Prefixes & Futures
    ↓
GRU Encoding (both prefix and future)
    ↓
Compute Similarity Matrix
    ↓
Weighted InfoNCE Loss (with timestep weighting)
    ↓
Backprop & Update Encoder
    ↓
Compute Top-1 / Top-5 Metrics
```

---

## Reference

This implementation follows the project specification in `GRU_implementation.md`. Key design decisions are documented there.

---

## Next Steps

- **Larger datasets**: Scale to more trajectories
- **Richer features**: Add visual features, semantic maps, or language instructions
- **Downstream tasks**: Use learned embeddings for planning, prediction, or navigation
- **Negative mining**: Implement hard negative mining to speed up learning
- **Multi-scale training**: Train on variable future lengths simultaneously
