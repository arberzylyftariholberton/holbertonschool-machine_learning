# Deep Convolutional Architectures (Inception, ResNet, DenseNet)

This project implements several important **deep convolutional neural network (CNN) architectures** using **TensorFlow / Keras**.

The purpose of the project is to understand how modern deep CNN architectures are constructed by implementing their **core building blocks** manually.

The architectures implemented in this project include:

- **GoogLeNet / Inception**
- **ResNet**
- **DenseNet**

These architectures introduced major improvements in deep learning by solving problems such as:

- inefficient feature extraction
- vanishing gradients
- redundant feature learning

---

## Project Structure

```
supervised_learning/
└── deep_cnns/
    ├── 0-inception_block.py
    ├── 1-inception_network.py
    ├── 2-identity_block.py
    ├── 3-projection_block.py
    ├── 4-resnet50.py
    ├── 5-dense_block.py
    ├── 6-transition_layer.py
    ├── 7-densenet121.py
    └── README.md
```

---

## Requirements

- Python 3
- TensorFlow / Keras

The project uses the following import restriction:

```python
from tensorflow import keras as K
```

All models use:

- He normal initialization
- Batch Normalization
- ReLU activation

---

## Implemented Architectures

The project gradually builds complete CNN architectures by first implementing the core blocks used in each network.

### 0. Inception Block

**File:** `0-inception_block.py`

This task implements an Inception block from the paper:

> *Going Deeper with Convolutions* (2014) — GoogLeNet.

The Inception block processes the input through multiple parallel branches:

- 1x1 convolution
- 1x1 → 3x3 convolution
- 1x1 → 5x5 convolution
- 3x3 max pooling → 1x1 convolution

All outputs are concatenated along the channel axis.

This design allows the network to extract features at multiple spatial scales simultaneously while keeping computational cost manageable.

---

### 1. Inception Network

**File:** `1-inception_network.py`

This task builds the full **GoogLeNet / Inception** architecture.

The network contains:

- initial convolution layers
- max pooling layers
- multiple stacked Inception modules
- global average pooling
- softmax classification layer

The key idea of GoogLeNet is to use parallel filters of different sizes to improve feature extraction efficiency.

---

### 2. Identity Block (ResNet)

**File:** `2-identity_block.py`

Implements the identity block from the paper:

> *Deep Residual Learning for Image Recognition* (2015).

This block is used when the input and output dimensions are the same.

**Structure:**

```
Input
 │
1x1 Conv
 │
3x3 Conv
 │
1x1 Conv
 │
Add shortcut connection
 │
ReLU
```

The shortcut connection allows the network to learn residual mappings, which helps train very deep neural networks.

---

### 3. Projection Block (ResNet)

**File:** `3-projection_block.py`

Implements the projection block from ResNet.

Unlike the identity block, the projection block is used when:

- spatial dimensions change
- the number of filters changes

The shortcut connection therefore uses a 1x1 convolution with stride to match dimensions.

**Structure:**

```
Main Path:
  1x1 Conv (stride s)
  3x3 Conv
  1x1 Conv

Shortcut Path:
  1x1 Conv (stride s)

Outputs are added together and passed through ReLU.
```

---

### 4. ResNet-50

**File:** `4-resnet50.py`

This task constructs the **ResNet-50** architecture.

ResNet-50 is composed of:

- initial convolution layer
- max pooling
- stacks of residual blocks
- global average pooling
- softmax classifier

The residual block pattern follows the stages:

```
Conv2_x → 3 blocks
Conv3_x → 4 blocks
Conv4_x → 6 blocks
Conv5_x → 3 blocks
```

Residual connections allow training of networks much deeper than traditional CNNs.

---

### 5. Dense Block (DenseNet)

**File:** `5-dense_block.py`

Implements a DenseNet dense block from the paper:

> *Densely Connected Convolutional Networks* (2017).

Each layer inside the dense block receives the concatenated output of all previous layers.

Each layer performs:

```
BatchNorm → ReLU → 1x1 Conv (bottleneck layer)
BatchNorm → ReLU → 3x3 Conv
```

This structure improves:

- feature reuse
- gradient flow
- parameter efficiency

Each new layer adds `growth_rate` feature maps.

---

### 6. Transition Layer (DenseNet)

**File:** `6-transition_layer.py`

Implements the transition layer used between dense blocks.

**Purpose:**

- reduce spatial dimensions
- compress the number of feature maps

**Structure:**

```
BatchNorm → ReLU → 1x1 Conv → Average Pooling
```

**Compression factor:**

```python
nb_filters = int(nb_filters * compression)
```

This follows the DenseNet-C design.

---

### 7. DenseNet-121

**File:** `7-densenet121.py`

Builds the complete **DenseNet-121** architecture.

The network consists of:

```
Initial Conv + Pool

Dense Block  (6 layers)  → Transition Layer
Dense Block  (12 layers) → Transition Layer
Dense Block  (24 layers) → Transition Layer
Dense Block  (16 layers)

Global Average Pooling → Softmax classifier
```

The growth rate controls how many feature maps each layer adds.

DenseNet reduces parameters while improving gradient propagation.

---

## Key Concepts

### Inception

Processes the same input using multiple filter sizes in parallel to capture both small and large features.

**Benefits:**

- multi-scale feature extraction
- efficient computation

### ResNet

Introduces skip connections:

```
F(x) + x
```

Instead of learning a direct mapping, the network learns a residual mapping.

**Benefits:**

- solves vanishing gradient problems
- enables very deep networks (50, 101, 152 layers)

### DenseNet

Connects each layer to all previous layers:

```
x0 → x1 → x2 → x3
 ↘    ↘    ↘
```

**Benefits:**

- feature reuse
- stronger gradient flow
- fewer parameters

---

## Example Usage

### Running an Identity Block

```python
from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block

X = K.Input(shape=(224, 224, 256))
Y = identity_block(X, [64, 64, 256])

model = K.models.Model(inputs=X, outputs=Y)
model.summary()
```

### Running ResNet-50

```python
resnet50 = __import__('4-resnet50').resnet50

model = resnet50()
model.summary()
```

### Running DenseNet-121

```python
densenet121 = __import__('7-densenet121').densenet121

model = densenet121(32, 0.5)
model.summary()
```

---

## Total Concepts Practiced

This project reinforces understanding of:

- advanced CNN architectures
- residual learning
- dense connectivity
- bottleneck layers
- model modularization
- TensorFlow / Keras functional API

---

## Author
- Arber Zylyftari
- Holberton School Machine Learning Specialization