# Convolutions and Pooling

Implementation of convolution and pooling operations for image processing and convolutional neural networks (CNNs). All implementations use NumPy with optimized vectorization (maximum 2-3 loops).

## Overview

This project implements fundamental CNN operations from scratch:

- **Valid Convolution**: No padding
- **Same Convolution**: Output matches input size
- **Custom Padding**: User-defined padding
- **Stride Support**: Downsampling via stride
- **Multi-channel Convolution**: RGB/color images
- **Multi-kernel Convolution**: Multiple feature maps
- **Pooling**: Max and average pooling

## Requirements

```bash
pip install numpy matplotlib
```

- Python 3.7+
- NumPy 1.19+
- Matplotlib (for visualization)

## Project Structure

```
convolutions_and_pooling/
├── 0-convolve_grayscale_valid.py    # Valid convolution
├── 1-convolve_grayscale_same.py     # Same convolution
├── 2-convolve_grayscale_padding.py  # Custom padding
├── 3-convolve_grayscale.py          # Full convolution (padding + stride)
├── 4-convolve_channels.py           # Channel convolution
├── 5-convolve.py                    # Multi-kernel convolution
└── 6-pool.py                        # Pooling operations
```

## Core Concepts

### Output Size Formulas

```
Valid:  oh = h - kh + 1
Same:   oh = h (when stride=1)
Custom: oh = (h + 2*ph - kh) // sh + 1
```

### Key Principles

**Convolution:**
- Slides kernel over image
- Multiplies and sums overlapping values
- Detects features (edges, patterns)

**Pooling:**
- Downsamples spatial dimensions
- Max: Takes maximum value
- Average: Takes mean value
- Keeps channel dimension

## Tasks

### Task 0: Valid Convolution

No padding, output smaller than input.

```python
def convolve_grayscale_valid(images, kernel):
    """
    Args:
        images: (m, h, w) - grayscale images
        kernel: (kh, kw) - convolution kernel
    Returns:
        (m, h-kh+1, w-kw+1) - convolved images
    """
```

**Example:**
```
Input:  (50000, 28, 28)
Kernel: (3, 3)
Output: (50000, 26, 26)  # Lost 2 pixels per side
```

### Task 1: Same Convolution

Adds padding to maintain input size.

```python
def convolve_grayscale_same(images, kernel):
    """Output same size as input (when stride=1)"""
```

**Padding calculation:**
```python
ph = kh // 2
pw = kw // 2
```

**Example:**
```
Input:  (50000, 28, 28)
Kernel: (3, 3)
Padding: (1, 1)
Output: (50000, 28, 28)  # Same size!
```

### Task 2: Custom Padding

User-defined padding amounts.

```python
def convolve_grayscale_padding(images, kernel, padding):
    """
    Args:
        padding: (ph, pw) - padding for height/width
    """
```

**Example:**
```
Input:  (50000, 28, 28)
Kernel: (3, 3)
Padding: (2, 4)
Output: (50000, 30, 34)  # Larger than input!
```

### Task 3: Full Convolution

Combines padding modes with stride support.

```python
def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Args:
        padding: 'same', 'valid', or (ph, pw)
        stride: (sh, sw) - stride for height/width
    """
```

**With stride:**
```
Input:  (50000, 28, 28)
Kernel: (3, 3)
Padding: 'valid'
Stride: (2, 2)
Output: (50000, 13, 13)  # Half size due to stride
```

### Task 4: Channel Convolution

Handles multi-channel images (RGB).

```python
def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Args:
        images: (m, h, w, c) - images with channels
        kernel: (kh, kw, c) - kernel matches channels
    Returns:
        (m, oh, ow) - single output (channels collapsed)
    """
```

**Key:** Sums over ALL channels → one output value per position.

**Example:**
```
Input:  (10000, 32, 32, 3)  # RGB
Kernel: (3, 3, 3)
Output: (10000, 30, 30)     # No channels!
```

### Task 5: Multi-Kernel Convolution

Multiple kernels create multiple output channels.

```python
def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Args:
        kernels: (kh, kw, c, nc) - nc kernels
    Returns:
        (m, oh, ow, nc) - nc output channels
    """
```

**Each kernel → one output channel**

**Example:**
```
Input:   (10000, 32, 32, 3)
Kernels: (3, 3, 3, 3)  # 3 kernels
Output:  (10000, 30, 30, 3)  # 3 output channels
```

### Task 6: Pooling

Downsampling via max or average pooling.

```python
def pool(images, kernel_shape, stride, mode='max'):
    """
    Args:
        kernel_shape: (kh, kw) - pooling window
        stride: (sh, sw)
        mode: 'max' or 'avg'
    Returns:
        (m, oh, ow, c) - pooled (channels preserved)
    """
```

**Max vs Average:**
```
Window:     Max:    Avg:
[1, 3]      6       3.75
[5, 6]
```

**Example:**
```
Input:  (10000, 32, 32, 3)
Pool:   (2, 2), stride (2, 2)
Output: (10000, 16, 16, 3)  # Half size, keeps channels
```

## Usage Examples

### Basic Convolution

```python
import numpy as np

# Load MNIST
dataset = np.load('MNIST.npz')
images = dataset['X_train']  # (50000, 28, 28)

# Edge detection kernel
kernel = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]])

# Valid convolution
output = convolve_grayscale_valid(images, kernel)
# Shape: (50000, 26, 26)
```

### RGB Image Processing

```python
# Load color images
dataset = np.load('animals_1.npz')
images = dataset['data']  # (10000, 32, 32, 3)

# Sharpen kernel (3x3x3)
kernel = np.array([[[0, 0, 0], [-1, -1, -1], [0, 0, 0]],
                   [[-1, -1, -1], [5, 5, 5], [-1, -1, -1]],
                   [[0, 0, 0], [-1, -1, -1], [0, 0, 0]]])

output = convolve_channels(images, kernel, padding='valid')
# Shape: (10000, 30, 30)
```

### CNN Layer Simulation

```python
# Multi-kernel convolution (like Conv2D layer)
kernels = np.random.randn(3, 3, 3, 8)  # 8 filters

output = convolve(images, kernels, padding='same', stride=(1, 1))
# Shape: (10000, 32, 32, 8)  # 8 feature maps

# Pooling layer
pooled = pool(output, (2, 2), (2, 2), mode='max')
# Shape: (10000, 16, 16, 8)  # Downsampled
```

## Implementation Details

### Vectorization Strategy

All implementations use NumPy's vectorization to process all images simultaneously:

```python
# Process ALL m images at once
for i in range(oh):
    for j in range(ow):
        patch = images[:, i:i+kh, j:j+kw]  # (m, kh, kw)
        output[:, i, j] = np.sum(patch * kernel, axis=(1, 2))
```

### Padding

```python
# Pad only spatial dimensions, not image/channel dims
padded = np.pad(images,
                ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                mode='constant')
```

### Stride

```python
# Stride implemented via indexing
patch = images[:, i*sh:i*sh+kh, j*sw:j*sw+kw]
```

## Performance Notes

| Operation | Complexity | Typical Use |
|-----------|------------|-------------|
| Valid Conv | O(m × oh × ow × kh × kw) | Feature extraction |
| Same Conv | O(m × h × w × kh × kw) | Preserve dimensions |
| Multi-kernel | O(m × oh × ow × kh × kw × nc) | CNN layers |
| Max Pool | O(m × oh × ow × kh × kw × c) | Downsampling |

## Common Use Cases

**Edge Detection:**
```python
kernel = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]])
```

**Blur:**
```python
kernel = np.ones((3, 3)) / 9
```

**Sharpen:**
```python
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
```

## Comparison: Convolution vs Pooling

| Aspect | Convolution | Pooling |
|--------|-------------|---------|
| Purpose | Feature detection | Downsampling |
| Weights | Learned | None |
| Channels | Can change | Preserved |
| Operation | Multiply + sum | Max or mean |

## Repository

- **Repository**: `holbertonschool-machine_learning`
- **Directory**: `math/convolutions_and_pooling`
- **Language**: Python 3.7+

---

**Author**: Arber Zylyftari - Holberton School ML Specialization  
**Version**: 1.0.0  
**Last Updated**: February 2026