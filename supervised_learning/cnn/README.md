# Convolutional Neural Networks (CNN)

Implementation of convolutional and pooling layer operations from scratch
using NumPy, plus a modified LeNet-5 architecture built with Keras/TensorFlow.

## Overview

This project covers the core building blocks of CNNs:

- **Forward propagation** through convolutional layers (valid/same padding, configurable stride)
- **Forward propagation** through pooling layers (max and average pooling)
- **Backpropagation** through convolutional layers (gradients w.r.t. inputs, weights, biases)
- **Backpropagation** through pooling layers (gradient routing for max/avg)
- **LeNet-5** modified architecture for 28x28 grayscale image classification (10 classes)

## Project Structure

```text
cnn/
├── 0-conv_forward.py      # Convolutional layer forward pass
├── 1-pool_forward.py      # Pooling layer forward pass
├── 2-conv_backward.py     # Convolutional layer backward pass
├── 3-pool_backward.py     # Pooling layer backward pass
├── 5-lenet5.py            # Modified LeNet-5 (Keras)
├── 0-main.py
├── 1-main.py
├── 2-main.py
├── 3-main.py
├── 5-main.py
└── README.md
```

## Tasks

| File | Function | Description |
|---|---|---|
| `0-conv_forward.py` | `conv_forward` | Forward pass over a convolutional layer |
| `1-pool_forward.py` | `pool_forward` | Forward pass over a pooling layer |
| `2-conv_backward.py` | `conv_backward` | Backpropagation through a convolutional layer |
| `3-pool_backward.py` | `pool_backward` | Backpropagation through a pooling layer |
| `5-lenet5.py` | `lenet5` | Modified LeNet-5 compiled Keras model |

## Key Concepts

### Convolutional Layer

A convolutional layer slides one or more kernels across the input volume.
The output size is determined by the kernel size, padding, and stride:

- **Valid padding**: no padding, output shrinks
- **Same padding**: zero-pad so output spatial dimensions match the input

### Pooling Layer

Pooling reduces the spatial dimensions of the feature maps:

- **Max pooling**: takes the maximum value in each window
- **Average pooling**: takes the mean value in each window

### LeNet-5 Architecture (modified)

```
Input (m, 28, 28, 1)
  → Conv2D(6, 5x5, relu, same)
  → MaxPool2D(2x2)
  → Conv2D(16, 5x5, relu, valid)
  → MaxPool2D(2x2)
  → Flatten
  → Dense(120, relu)
  → Dense(84, relu)
  → Dense(10, softmax)
```

Compiled with Adam optimizer and categorical cross-entropy loss.

## Requirements

- Python 3.9
- NumPy 1.25.2
- TensorFlow 2.15

## Usage

```bash
python3 0-main.py    # Test convolutional forward pass
python3 1-main.py    # Test pooling forward pass
python3 2-main.py    # Test convolutional backward pass
python3 3-main.py    # Test pooling backward pass
python3 5-main.py    # Test LeNet-5 model summary
```

## Repository

- **Repository**: `holbertonschool-machine_learning`
- **Directory**: `supervised_learning/cnn`
- **Language**: Python

---

**Author**: Arber Zylyftari
