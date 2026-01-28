# Keras Neural Network Implementation

A comprehensive toolkit for building, training, and deploying neural networks using TensorFlow Keras with industry-standard practices.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Core Components](#core-components)
- [API Reference](#api-reference)
- [Usage Examples](#usage-examples)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

This project provides a complete neural network toolkit featuring:

- **Model Building**: Sequential and Functional API implementations
- **Advanced Training**: Early stopping, learning rate decay, model checkpointing
- **Model Persistence**: Save/load complete models, weights, or configurations
- **Production Ready**: Evaluation and prediction pipelines

## Requirements

```bash
pip install tensorflow numpy
```

- Python 3.7+
- TensorFlow 2.x
- NumPy 1.19+

## Project Structure

```
├── 0-sequential.py    # Sequential API model builder
├── 1-input.py         # Functional API model builder
├── 2-optimize.py      # Model optimization (Adam, loss, metrics)
├── 3-one_hot.py       # One-hot encoding utility
├── 4-train.py         # Basic training
├── 5-train.py         # Training with validation
├── 6-train.py         # Training with early stopping
├── 7-train.py         # Training with learning rate decay
├── 8-train.py         # Training with model checkpointing
├── 9-model.py         # Full model save/load
├── 10-weights.py      # Weights save/load
├── 11-config.py       # Configuration save/load (JSON)
├── 12-test.py         # Model evaluation
└── 13-predict.py      # Model prediction
```

## Quick Start

```python
import numpy as np
from 1-input import build_model
from 2-optimize import optimize_model
from 3-one_hot import one_hot
from 8-train import train_model
from 12-test import test_model
from 13-predict import predict

# Load data
datasets = np.load('MNIST.npz')
X_train = datasets['X_train'].reshape(-1, 784) / 255.0
Y_train = one_hot(datasets['Y_train'])
X_test = datasets['X_test'].reshape(-1, 784) / 255.0
Y_test = one_hot(datasets['Y_test'])

# Build and compile model
model = build_model(784, [256, 256, 10], ['relu', 'relu', 'softmax'], 0.001, 0.95)
optimize_model(model, alpha=0.001, beta1=0.9, beta2=0.999)

# Train with all features
history = train_model(
    model, X_train, Y_train, batch_size=64, epochs=100,
    validation_data=(X_test, Y_test), early_stopping=True, patience=5,
    learning_rate_decay=True, alpha=0.001, save_best=True, filepath='best.keras'
)

# Evaluate and predict
loss, accuracy = test_model(model, X_test, Y_test)
predictions = predict(model, X_test[:10])
```

## Core Components

### 1. Model Building

**Sequential API (0-sequential.py)**
```python
model = build_model(nx, layers, activations, lambtha, keep_prob)
```
- Simple linear stack of layers
- L2 regularization and dropout included

**Functional API (1-input.py)**
```python
model = build_model(nx, layers, activations, lambtha, keep_prob)
```
- Supports complex architectures
- Better for multi-input/output models

### 2. Model Optimization (2-optimize.py)

```python
optimize_model(network, alpha, beta1, beta2)
```
Configures Adam optimizer with categorical crossentropy loss and accuracy metric.

### 3. Data Preprocessing (3-one_hot.py)

```python
labels_oh = one_hot(labels, classes=None)
```
Converts integer labels to one-hot encoded format.

### 4. Training Pipeline

Progressive training capabilities across modules 4-8:

**Basic Training (4-train.py)**
```python
train_model(network, data, labels, batch_size, epochs, verbose, shuffle)
```

**With Validation (5-train.py)**
```python
train_model(..., validation_data=(X_val, Y_val))
```

**With Early Stopping (6-train.py)**
```python
train_model(..., early_stopping=True, patience=3)
```

**With Learning Rate Decay (7-train.py)**
```python
train_model(..., learning_rate_decay=True, alpha=0.001, decay_rate=1)
```
Formula: `lr = alpha / (1 + decay_rate * epoch)`

**With Model Checkpointing (8-train.py)**
```python
train_model(..., save_best=True, filepath='model.keras')
```

### 5. Model Persistence

**Full Model (9-model.py)**
```python
save_model(network, filename)           # Architecture + weights + optimizer
loaded_model = load_model(filename)
```

**Weights Only (10-weights.py)**
```python
save_weights(network, filename, save_format='keras')
load_weights(network, filename)         # Requires existing architecture
```

**Configuration (11-config.py)**
```python
save_config(network, filename)          # Architecture only (JSON)
model = load_config(filename)           # Random weights
```

### 6. Evaluation & Prediction

**Testing (12-test.py)**
```python
loss, accuracy = test_model(network, data, labels, verbose=True)
```

**Prediction (13-predict.py)**
```python
predictions = predict(network, data, verbose=False)
predicted_classes = np.argmax(predictions, axis=1)
```

## API Reference

### build_model(nx, layers, activations, lambtha, keep_prob)
**Parameters:**
- `nx`: Number of input features
- `layers`: List of neurons per layer
- `activations`: List of activation functions
- `lambtha`: L2 regularization parameter
- `keep_prob`: Dropout keep probability (0-1)

**Returns:** Keras Model

### optimize_model(network, alpha, beta1, beta2)
**Parameters:**
- `alpha`: Learning rate
- `beta1`: First moment decay (typically 0.9)
- `beta2`: Second moment decay (typically 0.999)

### train_model(network, data, labels, batch_size, epochs, ...)
**Common Parameters:**
- `validation_data`: Tuple (X_val, Y_val)
- `early_stopping`: Enable early stopping
- `patience`: Epochs with no improvement before stopping
- `learning_rate_decay`: Enable inverse time decay
- `save_best`: Save best model during training
- `filepath`: Where to save best model

**Returns:** History object with training metrics

### test_model(network, data, labels, verbose=True)
**Returns:** List [loss, accuracy]

### predict(network, data, verbose=False)
**Returns:** NumPy array of predictions (probabilities for classification)

## Usage Examples

### Complete Training Pipeline

```python
# Build model
model = build_model(784, [512, 256, 10], ['relu', 'relu', 'softmax'], 0.0001, 0.95)
optimize_model(model, 0.001, 0.9, 0.999)

# Train with all features
history = train_model(
    model, X_train, Y_train, 
    batch_size=64, epochs=100,
    validation_data=(X_valid, Y_valid),
    early_stopping=True, patience=5,
    learning_rate_decay=True, alpha=0.001,
    save_best=True, filepath='best_model.keras'
)

# Evaluate
test_loss, test_acc = test_model(model, X_test, Y_test)
print(f"Test Accuracy: {test_acc:.4f}")
```

### Transfer Learning

```python
from 9-model import load_model
from 10-weights import save_weights, load_weights

# Load pretrained model and save weights
pretrained = load_model('pretrained.keras')
save_weights(pretrained, 'pretrained_weights.keras')

# Create new model and load weights
new_model = build_model(784, [256, 256, 10], ['relu', 'relu', 'softmax'], 0.001, 0.95)
load_weights(new_model, 'pretrained_weights.keras')

# Fine-tune on new data
optimize_model(new_model, 0.0001, 0.9, 0.999)  # Lower learning rate
train_model(new_model, X_new, Y_new, batch_size=32, epochs=10)
```

### Model Architecture Comparison

```python
from 11-config import save_config, load_config

# Save architecture template
save_config(base_model, 'architecture.json')

# Create multiple models with same architecture
models = [load_config('architecture.json') for _ in range(3)]
learning_rates = [0.001, 0.0001, 0.01]

# Train and compare
for model, lr in zip(models, learning_rates):
    optimize_model(model, lr, 0.9, 0.999)
    train_model(model, X_train, Y_train, 32, 20, validation_data=(X_val, Y_val))
    loss, acc = test_model(model, X_test, Y_test, verbose=False)
    print(f"LR={lr}: Accuracy={acc:.4f}")
```

## Best Practices

### Data Preprocessing
```python
# Normalize features
X = X / 255.0  # For images [0-255] → [0-1]

# One-hot encode labels
Y = one_hot(Y, classes=10)

# Split data: 70% train, 15% validation, 15% test
```

### Model Architecture
- **Layer sizes**: Gradually decrease (e.g., 512 → 256 → 128 → 10)
- **Hidden activations**: ReLU (default), tanh (older)
- **Output activation**: Softmax (classification), Linear (regression)
- **Regularization**: L2=0.0001-0.001, dropout=0.8-0.95

### Training
- **Batch size**: 32-64 (small data), 128-256 (large data)
- **Learning rate**: Start with 0.001 for Adam
- **Always use validation data** to monitor overfitting
- **Enable early stopping** to save time and prevent overfitting
- **Use model checkpointing** to save best model

### Model Selection
- **Full model**: Production deployment
- **Weights only**: Checkpointing during training
- **Config only**: Architecture sharing, version control

## Troubleshooting

### Model Not Learning
- Check learning rate (try 0.001, 0.0001, 0.01)
- Verify data preprocessing (normalization, one-hot encoding)
- Ensure sufficient model capacity

### Overfitting (Train acc >> Val acc)
- Increase dropout: `keep_prob=0.5-0.8`
- Increase L2: `lambtha=0.01`
- Use early stopping
- Get more training data

### Underfitting (Both accuracies low)
- Increase model size (more layers/neurons)
- Reduce regularization
- Train longer

### Memory Errors
- Reduce batch size
- Reduce model size
- Use float32 instead of float64

### Common Errors

**Shape mismatch:**
```python
X = X.reshape(X.shape[0], -1)  # Flatten
Y = one_hot(Y, classes=10)      # One-hot encode
```

**No gradients:**
```python
optimize_model(model, 0.001, 0.9, 0.999)  # Must compile before training
```

## Performance Tips

```python
# Use appropriate batch sizes (powers of 2 for GPU)
batch_size = 64

# Clear session between experiments
from tensorflow.keras import backend as K
K.clear_session()

# Use float32 for efficiency
X_train = X_train.astype('float32')
```

## Contributing

Contributions welcome! Please:
1. Follow PEP 8 style guidelines
2. Add tests for new features
3. Update documentation

## License

MIT License - See LICENSE file for details

---

**Version**: 1.0.0  
**Last Updated**: January 2026  
**Status**: Active Development