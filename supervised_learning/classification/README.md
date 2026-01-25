# Supervised Learning - Classification

This project implements binary and multiclass classification using neural networks in Python with NumPy. The tasks progress from a single neuron to deep neural networks with multiple layers and various activation functions.

## Table of Contents

- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Tasks](#tasks)
  - [Single Neuron (Tasks 0-7)](#single-neuron-tasks-0-7)
  - [Neural Network (Tasks 8-15)](#neural-network-tasks-8-15)
  - [Deep Neural Network (Tasks 16-28)](#deep-neural-network-tasks-16-28)
- [Usage Examples](#usage-examples)
- [Key Concepts](#key-concepts)

## Project Overview

This project demonstrates the implementation of classification algorithms from scratch, including:

- Binary classification using logistic regression
- Multiclass classification using softmax
- Single neuron and multi-layer neural networks
- Forward propagation and backpropagation
- Gradient descent optimization
- Different activation functions (sigmoid, tanh, softmax)
- Model persistence using pickle
- One-hot encoding for multiclass problems

## Requirements

- Python 3.x
- NumPy
- Matplotlib (for visualization)

## Tasks

### Single Neuron (Tasks 0-7)

#### Task 0: Neuron
**File:** `0-neuron.py`

Creates a `Neuron` class that defines a single neuron performing binary classification with public instance attributes.

**Public Attributes:**
- `W`: Weights vector initialized using random normal distribution
- `b`: Bias initialized to 0
- `A`: Activated output (prediction) initialized to 0

#### Task 1: Privatize Neuron
**File:** `1-neuron.py`

Updates the `Neuron` class with private instance attributes and getter methods.

**Private Attributes:**
- `__W`: Weights vector
- `__b`: Bias
- `__A`: Activated output

#### Task 2: Neuron Forward Propagation
**File:** `2-neuron.py`

Adds forward propagation using sigmoid activation function:
```
σ(z) = 1 / (1 + e^(-z))
```

#### Task 3: Neuron Cost
**File:** `3-neuron.py`

Implements logistic regression cost calculation:
```
J = -(1/m) * Σ[y*log(a) + (1-y)*log(1-a)]
```

#### Task 4: Evaluate Neuron
**File:** `4-neuron.py`

Evaluates the neuron's predictions by returning predicted labels (0 or 1) and cost.

#### Task 5: Neuron Gradient Descent
**File:** `5-neuron.py`

Implements one step of gradient descent to update weights and bias.

#### Task 6: Train Neuron
**File:** `6-neuron.py`

Trains the neuron over multiple iterations using forward propagation and gradient descent.

#### Task 7: Upgrade Train Neuron
**File:** `7-neuron.py`

Enhances training with:
- Verbose output showing cost at intervals
- Graphical plotting of training cost
- Step parameter for output frequency

### Neural Network (Tasks 8-15)

#### Task 8: NeuralNetwork
**File:** `8-neural_network.py`

Creates a `NeuralNetwork` class with one hidden layer for binary classification.

**Public Attributes:**
- `W1`: Hidden layer weights
- `b1`: Hidden layer bias
- `A1`: Hidden layer activation
- `W2`: Output neuron weights
- `b2`: Output neuron bias
- `A2`: Output neuron activation

#### Task 9: Privatize NeuralNetwork
**File:** `9-neural_network.py`

Updates `NeuralNetwork` with private attributes and getter methods.

#### Task 10: NeuralNetwork Forward Propagation
**File:** `10-neural_network.py`

Implements forward propagation through the hidden layer and output neuron using sigmoid activation.

#### Task 11: NeuralNetwork Cost
**File:** `11-neural_network.py`

Calculates the cost using logistic regression for the neural network.

#### Task 12: Evaluate NeuralNetwork
**File:** `12-neural_network.py`

Evaluates the neural network's predictions.

#### Task 13: NeuralNetwork Gradient Descent
**File:** `13-neural_network.py`

Implements backpropagation and gradient descent for the neural network.

#### Task 14: Train NeuralNetwork
**File:** `14-neural_network.py`

Trains the neural network over multiple iterations.

#### Task 15: Upgrade Train NeuralNetwork
**File:** `15-neural_network.py`

Adds verbose output and cost visualization to neural network training.

### Deep Neural Network (Tasks 16-28)

#### Task 16: DeepNeuralNetwork
**File:** `16-deep_neural_network.py`

Creates a `DeepNeuralNetwork` class with multiple layers.

**Public Attributes:**
- `L`: Number of layers
- `cache`: Dictionary storing intermediate values
- `weights`: Dictionary storing all weights and biases

**Weight Initialization:** He et al. method
```
W = np.random.randn(nodes, prev) * np.sqrt(2 / prev)
```

#### Task 17: Privatize DeepNeuralNetwork
**File:** `17-deep_neural_network.py`

Updates `DeepNeuralNetwork` with private attributes and getter methods.

#### Task 18: DeepNeuralNetwork Forward Propagation
**File:** `18-deep_neural_network.py`

Implements forward propagation through all layers using sigmoid activation.

#### Task 19: DeepNeuralNetwork Cost
**File:** `19-deep_neural_network.py`

Calculates cost using logistic regression for the deep network.

#### Task 20: Evaluate DeepNeuralNetwork
**File:** `20-deep_neural_network.py`

Evaluates predictions of the deep neural network.

#### Task 21: DeepNeuralNetwork Gradient Descent
**File:** `21-deep_neural_network.py`

Implements backpropagation through all layers using gradient descent.

#### Task 22: Train DeepNeuralNetwork
**File:** `22-deep_neural_network.py`

Trains the deep neural network over multiple iterations.

#### Task 23: Upgrade Train DeepNeuralNetwork
**File:** `23-deep_neural_network.py`

Adds verbose output and cost visualization to deep network training.

#### Task 24: One-Hot Encode
**File:** `24-one_hot_encode.py`

Converts numeric label vectors into one-hot matrices for multiclass classification.

```python
def one_hot_encode(Y, classes):
    """
    Converts numeric labels to one-hot encoding.
    
    Args:
        Y: numpy array of shape (m,) with numeric labels
        classes: number of classes
    
    Returns:
        One-hot matrix of shape (classes, m)
    """
```

#### Task 25: One-Hot Decode
**File:** `25-one_hot_decode.py`

Converts one-hot matrices back to numeric label vectors.

```python
def one_hot_decode(one_hot):
    """
    Converts one-hot matrix to numeric labels.
    
    Args:
        one_hot: numpy array of shape (classes, m)
    
    Returns:
        Array of shape (m,) with numeric labels
    """
```

#### Task 26: Persistence
**File:** `26-deep_neural_network.py`

Adds model persistence using pickle:
- `save(filename)`: Saves model to pickle file
- `load(filename)`: Loads model from pickle file (static method)

#### Task 27: Update for Multiclass
**File:** `27-deep_neural_network.py`

Updates `DeepNeuralNetwork` for multiclass classification:
- Uses **softmax** activation in output layer
- Uses **sigmoid** in hidden layers
- Implements **multiclass cross-entropy** cost

**Softmax Function:**
```
softmax(z_i) = e^(z_i) / Σ(e^(z_j))
```

**Multiclass Cross-Entropy:**
```
J = -(1/m) * Σ Σ [y_ij * log(a_ij)]
```

#### Task 28: All the Activations
**File:** `28-deep_neural_network.py`

Adds flexible activation functions for hidden layers:
- `activation='sig'`: Sigmoid activation
- `activation='tanh'`: Tanh activation

**Tanh Function:**
```
tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
```

**Tanh Derivative:**
```
tanh'(z) = 1 - tanh²(z)
```

## Usage Examples

### Training a Single Neuron

```python
#!/usr/bin/env python3
import numpy as np
Neuron = __import__('7-neuron').Neuron

# Load data
lib_train = np.load('data/Binary_Train.npz')
X_train = lib_train['X']
Y_train = lib_train['Y']

# Create and train neuron
neuron = Neuron(X_train.shape[0])
A, cost = neuron.train(X_train, Y_train, iterations=3000)

print("Final cost:", cost)
print("Accuracy:", np.sum(A == Y_train) / Y_train.shape[1])
```

### Training a Deep Neural Network

```python
#!/usr/bin/env python3
import numpy as np
DeepNeuralNetwork = __import__('23-deep_neural_network').DeepNeuralNetwork

# Load data
lib = np.load('data/Binary_Train.npz')
X_train = lib['X']
Y_train = lib['Y']

# Create deep network
deep = DeepNeuralNetwork(X_train.shape[0], [5, 3, 1])

# Train with visualization
A, cost = deep.train(X_train, Y_train, iterations=5000, 
                     verbose=True, graph=True, step=100)
```

### Multiclass Classification

```python
#!/usr/bin/env python3
import numpy as np
DeepNeuralNetwork = __import__('28-deep_neural_network').DeepNeuralNetwork
one_hot_encode = __import__('24-one_hot_encode').one_hot_encode

# Load data
lib = np.load('data/MNIST.npz')
X_train = lib['X_train']
Y_train = lib['Y_train']

# One-hot encode labels
Y_train_oh = one_hot_encode(Y_train, 10)

# Create and train network
deep = DeepNeuralNetwork(X_train.shape[0], [128, 64, 10], 
                         activation='tanh')
deep.train(X_train, Y_train_oh, iterations=1000)

# Save model
deep.save('model.pkl')

# Load model later
loaded = DeepNeuralNetwork.load('model.pkl')
```

## Key Concepts

### Forward Propagation
The process of passing input data through the network layers to produce predictions.

For each layer l:
```
Z[l] = W[l] · A[l-1] + b[l]
A[l] = activation(Z[l])
```

### Backpropagation
The process of computing gradients by propagating errors backwards through the network.

For output layer:
```
dZ[L] = A[L] - Y
```

For hidden layers (sigmoid):
```
dZ[l] = W[l+1]ᵀ · dZ[l+1] * A[l] * (1 - A[l])
```

For hidden layers (tanh):
```
dZ[l] = W[l+1]ᵀ · dZ[l+1] * (1 - A[l]²)
```

### Gradient Descent Update
```
W[l] = W[l] - α * dW[l]
b[l] = b[l] - α * db[l]
```

Where:
```
dW[l] = (1/m) * dZ[l] · A[l-1]ᵀ
db[l] = (1/m) * Σ(dZ[l])
```

### Activation Functions

**Sigmoid:**
- Range: (0, 1)
- Use: Binary classification output
- Derivative: σ(z) * (1 - σ(z))

**Tanh:**
- Range: (-1, 1)
- Use: Hidden layers (often faster than sigmoid)
- Derivative: 1 - tanh²(z)

**Softmax:**
- Range: (0, 1), sum = 1
- Use: Multiclass classification output
- Derivative: Combined with cross-entropy for simplified gradient

### Weight Initialization

**The Initialization:**
```
W = np.random.randn(n, n_prev) * np.sqrt(2 / n_prev)
```

Benefits:
- Prevents vanishing/exploding gradients
- Works well with ReLU and similar activations
- Maintains variance through layers

---

**Author:** Arber Zylyftari  
**School:** Holberton School  
**Project:** Supervised Learning - Classification