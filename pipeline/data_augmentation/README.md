# TensorFlow Image Augmentation

Implementation of basic image augmentation operations using **TensorFlow**.  
These functions apply common transformations used in computer vision and deep learning pipelines.

All operations use built-in `tf.image` methods and operate on 3D image tensors.

---

## Overview

This project implements fundamental image augmentation techniques:

- **Horizontal Flip**
- **Random Crop**
- **Rotation (90° counter-clockwise)**
- **Random Contrast Adjustment**
- **Random Brightness Adjustment**
- **Hue Adjustment**

These transformations are commonly used to increase dataset diversity and improve model generalization.

---

## Requirements

```bash
pip install tensorflow tensorflow-datasets matplotlib
```

- Python 3.7+
- TensorFlow 2.x
- TensorFlow Datasets
- Matplotlib (for visualization)

---

## Project Structure

```
tensorflow_image_augmentation/
├── 0-flip.py          # Horizontal flip
├── 1-crop.py          # Random crop
├── 2-rotate.py        # 90° rotation
├── 3-contrast.py      # Random contrast adjustment
├── 4-brightness.py    # Random brightness adjustment
└── 5-hue.py           # Hue adjustment
```

---

## Core Concepts

### Image Tensor Format

All functions expect:

```
image: tf.Tensor of shape (height, width, channels)
```

- Height → Image height  
- Width → Image width  
- Channels → Typically 3 (RGB)

Example shape:

```
(300, 500, 3)
```

---

# Tasks

---

## Task 0: Horizontal Flip

Flips an image horizontally (mirror effect).

```python
def flip_image(image):
    """
    Args:
        image: 3D tf.Tensor (h, w, c)
    Returns:
        Flipped image
    """
```

**Effect Example:**

```
Before:  Dog facing right
After:   Dog facing left
```

---

## Task 1: Random Crop

Randomly crops a region from the image.

```python
def crop_image(image, size):
    """
    Args:
        image: 3D tf.Tensor
        size: (new_height, new_width, channels)
    Returns:
        Cropped image
    """
```

**Example:**

```
Original: (300, 500, 3)
Crop:     (200, 200, 3)
Output:   (200, 200, 3)
```

---

## Task 2: Rotation

Rotates the image 90 degrees counter-clockwise.

```python
def rotate_image(image):
    """
    Args:
        image: 3D tf.Tensor
    Returns:
        Rotated image
    """
```

**Example:**

```
Original shape: (300, 500, 3)
Rotated shape:  (500, 300, 3)
```

---

## Task 3: Random Contrast Adjustment

Randomly changes the contrast of an image.

```python
def change_contrast(image, lower, upper):
    """
    Args:
        image: 3D tf.Tensor
        lower: Minimum contrast factor
        upper: Maximum contrast factor
    Returns:
        Contrast-adjusted image
    """
```

Contrast factor is randomly chosen between `lower` and `upper`.

---

## Task 4: Random Brightness Adjustment

Randomly increases or decreases brightness.

```python
def change_brightness(image, max_delta):
    """
    Args:
        image: 3D tf.Tensor
        max_delta: Maximum brightness change
    Returns:
        Brightness-adjusted image
    """
```

Brightness is randomly adjusted within:

```
[-max_delta, +max_delta]
```

---

## Task 5: Hue Adjustment

Changes the hue of an image.

```python
def change_hue(image, delta):
    """
    Args:
        image: 3D tf.Tensor
        delta: Amount of hue shift
    Returns:
        Hue-adjusted image
    """
```

- Positive delta → shifts hue forward  
- Negative delta → shifts hue backward  

---

## Usage Example

```python
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from flip import flip_image

tf.random.set_seed(0)

doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)

for image, _ in doggies.shuffle(10).take(1):
    augmented = flip_image(image)
    plt.imshow(augmented)
    plt.show()
```

---

## Why Image Augmentation Matters

Without augmentation:
- Models may overfit
- Poor generalization

With augmentation:
- Increased dataset variability
- Improved robustness
- Better real-world performance

---

## TensorFlow Functions Used

| Task        | TensorFlow Function              |
|------------|----------------------------------|
| Flip       | `tf.image.flip_left_right`       |
| Crop       | `tf.image.random_crop`           |
| Rotate     | `tf.image.rot90`                 |
| Contrast   | `tf.image.random_contrast`       |
| Brightness | `tf.image.random_brightness`     |
| Hue        | `tf.image.adjust_hue`            |

---

## Repository

- **Repository**: `holbertonschool-machine_learning`
- **Directory**: `supervised_learning/data_augmentation`
- **Framework**: TensorFlow 2.x
- **Language**: Python 3.7+

---

**Author**: Arber Zylyftari – Holberton School ML Specialization  
**Version**: 1.0.0  
**Last Updated**: March 2026