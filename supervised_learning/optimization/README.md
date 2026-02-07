# Optimization Algorithms for Machine Learning

A comprehensive implementation of optimization algorithms and techniques for training neural networks, including normalization, gradient descent variants, learning rate scheduling, and batch normalization.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Core Concepts](#core-concepts)
- [Tasks](#tasks)
- [Usage Examples](#usage-examples)
- [Best Practices](#best-practices)
- [Resources](#resources)

## Overview

This project implements fundamental optimization algorithms used in deep learning, progressing from basic data preprocessing to advanced optimization techniques:

- **Data Normalization**: Standardization and preprocessing
- **Gradient Descent Variants**: Momentum, RMSProp, and Adam optimizers
- **Learning Rate Scheduling**: Inverse time decay strategies
- **Batch Normalization**: Internal covariate shift reduction

Each implementation includes both NumPy (from-scratch) and TensorFlow (production-ready) versions.

## Requirements

```bash
pip install numpy tensorflow matplotlib
```

- Python 3.7+
- NumPy 1.19+
- TensorFlow 2.x
- Matplotlib 3.x (for visualizations)

## Project Structure

```
optimization/
├── 0-norm_constants.py       # Calculate normalization constants
├── 1-normalize.py             # Normalize data (standardization)
├── 2-shuffle_data.py          # Shuffle dataset
├── 3-mini_batch.py            # Create mini-batches
├── 4-moving_average.py        # Exponentially weighted moving average
├── 5-momentum.py              # Momentum optimization (NumPy)
├── 6-momentum.py              # Momentum optimization (TensorFlow)
├── 7-RMSProp.py               # RMSProp optimization (NumPy)
├── 8-RMSProp.py               # RMSProp optimization (TensorFlow)
├── 9-Adam.py                  # Adam optimization (NumPy)
├── 10-Adam.py                 # Adam optimization (TensorFlow)
├── 11-learning_rate_decay.py  # Learning rate decay (NumPy)
├── 12-learning_rate_decay.py  # Learning rate decay (TensorFlow)
├── 13-batch_norm.py           # Batch normalization (NumPy)
└── 14-batch_norm.py           # Batch normalization (TensorFlow)
```

## Core Concepts

### 1. Data Preprocessing

**Normalization (Standardization)**
```python
mean, std = normalization_constants(X)
X_normalized = normalize(X, mean, std)
```
- Transforms data to zero mean and unit variance
- Essential for gradient descent convergence
- Apply same normalization to train/validation/test sets

**Data Shuffling**
```python
X_shuffled, Y_shuffled = shuffle_data(X, Y)
```
- Removes ordering bias
- Improves generalization
- Creates random mini-batches

### 2. Mini-Batch Gradient Descent

```python
batches = create_mini_batches(X, Y, batch_size=32)
for X_batch, Y_batch in batches:
    # Train on mini-batch
```
- Balances speed (batch GD) and stability (SGD)
- Enables efficient GPU utilization
- Standard batch sizes: 32, 64, 128, 256

### 3. Optimization Algorithms

**Momentum**
```
v = beta1 * v + (1 - beta1) * gradient
parameter = parameter - alpha * v
```
- Accelerates convergence in consistent directions
- Dampens oscillations
- Typical beta1: 0.9

**RMSProp**
```
s = beta2 * s + (1 - beta2) * gradient²
parameter = parameter - alpha * gradient / (√s + epsilon)
```
- Adaptive per-parameter learning rates
- Handles different feature scales
- Typical beta2: 0.9 or 0.999

**Adam (Momentum + RMSProp)**
```
v = beta1 * v + (1 - beta1) * gradient
s = beta2 * s + (1 - beta2) * gradient²
v_corrected = v / (1 - beta1^t)
s_corrected = s / (1 - beta2^t)
parameter = parameter - alpha * v_corrected / (√s_corrected + epsilon)
```
- Most popular optimizer
- Combines advantages of Momentum and RMSProp
- Includes bias correction
- Default hyperparameters work well: alpha=0.001, beta1=0.9, beta2=0.999

### 4. Learning Rate Scheduling

**Inverse Time Decay**
```
alpha = alpha_0 / (1 + decay_rate * floor(step / decay_steps))
```
- Larger learning rate early (fast progress)
- Smaller learning rate later (fine-tuning)
- Stepwise decay for stability

### 5. Batch Normalization

```
mean = mean(Z, axis=0)
variance = var(Z, axis=0)
Z_norm = (Z - mean) / √(variance + epsilon)
Z_output = gamma * Z_norm + beta
```
- Normalizes layer inputs
- Enables higher learning rates
- Acts as regularization
- Faster convergence

## Tasks

### Task 0-1: Normalization
Calculate mean/std and normalize data to zero mean and unit variance.

```python
from 0-norm_constants import normalization_constants
from 1-normalize import normalize

mean, std = normalization_constants(X_train)
X_train_norm = normalize(X_train, mean, std)
X_test_norm = normalize(X_test, mean, std)  # Use training stats!
```

### Task 2-3: Mini-Batch Creation
Shuffle data and create mini-batches for training.

```python
from 2-shuffle_data import shuffle_data
from 3-mini_batch import create_mini_batches

X_shuffled, Y_shuffled = shuffle_data(X, Y)
batches = create_mini_batches(X, Y, batch_size=64)
```

### Task 4: Moving Average
Calculate exponentially weighted moving average with bias correction.

```python
from 4-moving_average import moving_average

smoothed_data = moving_average(data, beta=0.9)
```

### Task 5-6: Momentum Optimization
Implement momentum in NumPy and TensorFlow.

```python
# NumPy
from 5-momentum import update_variables_momentum
W, dW_prev = update_variables_momentum(0.01, 0.9, W, dW, dW_prev)

# TensorFlow
from 6-momentum import create_momentum_op
optimizer = create_momentum_op(alpha=0.01, beta1=0.9)
```

### Task 7-8: RMSProp Optimization
Implement RMSProp in NumPy and TensorFlow.

```python
# NumPy
from 7-RMSProp import update_variables_RMSProp
W, dW_prev = update_variables_RMSProp(0.001, 0.9, 1e-8, W, dW, dW_prev)

# TensorFlow
from 8-RMSProp import create_RMSProp_op
optimizer = create_RMSProp_op(alpha=0.001, beta2=0.9, epsilon=1e-7)
```

### Task 9-10: Adam Optimization
Implement Adam optimizer in NumPy and TensorFlow.

```python
# NumPy
from 9-Adam import update_variables_Adam
W, v, s = update_variables_Adam(0.001, 0.9, 0.999, 1e-8, W, dW, v, s, t)

# TensorFlow
from 10-Adam import create_Adam_op
optimizer = create_Adam_op(alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7)
```

### Task 11-12: Learning Rate Decay
Implement inverse time decay in NumPy and TensorFlow.

```python
# NumPy
from 11-learning_rate_decay import learning_rate_decay
alpha = learning_rate_decay(alpha_init, decay_rate=1, global_step=step, decay_step=100)

# TensorFlow
from 12-learning_rate_decay import learning_rate_decay
lr_schedule = learning_rate_decay(alpha=0.1, decay_rate=1, decay_step=100)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
```

### Task 13-14: Batch Normalization
Implement batch normalization in NumPy and TensorFlow.

```python
# NumPy
from 13-batch_norm import batch_norm
Z_normalized = batch_norm(Z, gamma, beta, epsilon=1e-7)

# TensorFlow
from 14-batch_norm import create_batch_norm_layer
output = create_batch_norm_layer(prev, n=256, activation=tf.nn.relu)
```

## Usage Examples

### Complete Training Pipeline

```python
import numpy as np
import tensorflow as tf
from 1-normalize import normalize
from 0-norm_constants import normalization_constants
from 3-mini_batch import create_mini_batches
from 10-Adam import create_Adam_op

# 1. Load and preprocess data
X_train, Y_train = load_data()
mean, std = normalization_constants(X_train)
X_train = normalize(X_train, mean, std)

# 2. Build model
model = build_model()

# 3. Create optimizer with learning rate decay
from 12-learning_rate_decay import learning_rate_decay
lr_schedule = learning_rate_decay(0.001, 1, 1000)
optimizer = create_Adam_op(lr_schedule, 0.9, 0.999, 1e-7)

# 4. Training loop
for epoch in range(epochs):
    batches = create_mini_batches(X_train, Y_train, batch_size=64)
    
    for X_batch, Y_batch in batches:
        with tf.GradientTape() as tape:
            predictions = model(X_batch)
            loss = loss_fn(Y_batch, predictions)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

### Optimizer Comparison

```python
# Test different optimizers on same problem
optimizers = {
    'SGD': tf.keras.optimizers.SGD(0.01),
    'Momentum': create_momentum_op(0.01, 0.9),
    'RMSProp': create_RMSProp_op(0.001, 0.9, 1e-7),
    'Adam': create_Adam_op(0.001, 0.9, 0.999, 1e-7)
}

for name, optimizer in optimizers.items():
    model = build_model()
    history = train(model, optimizer, epochs=100)
    print(f"{name}: Final loss = {history.history['loss'][-1]:.4f}")
```

## Best Practices

### Data Preprocessing
```python
# Always normalize features
mean, std = normalization_constants(X_train)
X_train = normalize(X_train, mean, std)
X_test = normalize(X_test, mean, std)  # Use training statistics!

# Shuffle before creating batches
X, Y = shuffle_data(X, Y)
```

### Optimizer Selection

| Optimizer | When to Use | Typical Learning Rate |
|-----------|-------------|---------------------|
| **SGD** | Simple problems, well-tuned | 0.01 - 0.1 |
| **SGD + Momentum** | CNNs, image classification | 0.01 - 0.1 |
| **RMSProp** | RNNs, varying scales | 0.001 - 0.01 |
| **Adam** | General purpose, default choice | 0.0001 - 0.001 |

### Hyperparameter Guidelines

```python
# Adam (recommended defaults)
optimizer = create_Adam_op(
    alpha=0.001,      # or 0.0001 for fine-tuning
    beta1=0.9,        # momentum
    beta2=0.999,      # RMSProp (not 0.99!)
    epsilon=1e-7      # numerical stability
)

# Batch size
batch_size = 32   # Small datasets
batch_size = 64   # Standard choice
batch_size = 128  # Large datasets, GPUs

# Learning rate decay
lr_schedule = learning_rate_decay(
    alpha=0.1,
    decay_rate=1.0,
    decay_step=1000  # Decay every N steps
)
```

### Batch Normalization

```python
# Standard placement
Input → Dense → BatchNorm → Activation → Output

# Can disable Dense bias (BatchNorm's beta acts as bias)
Dense(units=256, use_bias=False) → BatchNorm → ReLU
```

## Algorithm Comparison

### Convergence Speed (MNIST Example)

| Optimizer | Cost after 1000 iterations | Relative Speed |
|-----------|---------------------------|----------------|
| Vanilla SGD | ~0.5 | Baseline |
| Momentum | 0.070 | 7x faster |
| RMSProp | 0.0003 | 1600x faster |
| Adam | 0.009 | 55x faster |

### Memory Requirements

| Algorithm | Extra Memory per Parameter |
|-----------|---------------------------|
| SGD | None |
| Momentum | 1x (velocity) |
| RMSProp | 1x (squared gradients) |
| Adam | 2x (velocity + squared gradients) |

## Common Pitfalls

❌ **Don't:**
- Use training statistics for test data normalization
- Forget to shuffle data before training
- Use batch size = 1 with batch normalization
- Apply learning rate decay with Adam (usually unnecessary)
- Use very large batch sizes without adjusting learning rate

✅ **Do:**
- Normalize all datasets with training statistics
- Shuffle data each epoch
- Use batch sizes that fit GPU memory (powers of 2)
- Start with Adam optimizer (good default)
- Monitor validation loss to detect overfitting

## Troubleshooting

### Loss not decreasing
- Check learning rate (try 0.001, 0.0001, 0.01)
- Verify data normalization
- Ensure correct gradient computation
- Try different optimizer (Adam is robust)

### Loss exploding
- Reduce learning rate
- Add gradient clipping
- Check for bugs in loss function
- Use batch normalization

### Slow convergence
- Increase learning rate
- Use momentum or Adam
- Add batch normalization
- Increase batch size

### Overfitting
- Add dropout/regularization
- Use early stopping
- Get more training data
- Reduce model complexity

## Resources

### Papers
- **Adam**: [Kingma & Ba, 2014](https://arxiv.org/abs/1412.6980)
- **Batch Normalization**: [Ioffe & Szegedy, 2015](https://arxiv.org/abs/1502.03167)
- **RMSProp**: Hinton's Coursera Lecture 6e

### Documentation
- [TensorFlow Optimizers](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers)
- [NumPy Documentation](https://numpy.org/doc/)

## Repository Information

- **Repository**: `holbertonschool-machine_learning`
- **Directory**: `supervised_learning/optimization`
- **Language**: Python 3.7+
- **Style**: PEP 8 compliant

## Author
Arber Zylyftari
Holberton School Machine Learning Specialization

---

**Version**: 1.0.0  
**Last Updated**: January 2026  
**Status**: Complete