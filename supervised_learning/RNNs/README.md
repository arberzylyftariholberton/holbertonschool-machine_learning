# Recurrent Neural Networks

Implementation of core recurrent neural network building blocks using
NumPy, including a simple RNN cell, forward propagation for shallow and
deep RNNs, a GRU cell, and an LSTM cell.

## Overview

This project covers the fundamental components used to build sequence
models from scratch:

- **Simple RNN Cell**: one recurrent unit with hidden state updates
- **RNN Forward Propagation**: runs a simple RNN across time steps
- **GRU Cell**: gated recurrent unit with update and reset gates
- **LSTM Cell**: long short-term memory unit with cell state
- **Deep RNN**: stacked recurrent layers across time

Each implementation uses only `numpy as np`, matching the task
constraints for the Holberton machine learning curriculum.

## Requirements

- Python 3.9
- NumPy 1.25.2
- Ubuntu 20.04 LTS
- `pycodestyle` 2.11.1

All files:

- start with `#!/usr/bin/env python3`
- end with a new line
- include module, class, and function documentation

## Project Structure

```text
RNNs/
├── 0-rnn_cell.py    # Simple RNN cell
├── 0-main.py        # Test file for task 0
├── 1-rnn.py         # Forward propagation for a simple RNN
├── 1-main.py        # Test file for task 1
├── 2-gru_cell.py    # GRU cell
├── 2-main.py        # Test file for task 2
├── 3-lstm_cell.py   # LSTM cell
├── 3-main.py        # Test file for task 3
├── 4-deep_rnn.py    # Forward propagation for a deep RNN
├── 4-main.py        # Test file for task 4
└── README.md
```

## Tasks

### Task 0: RNN Cell

Defines the `RNNCell` class with:

- weight matrix for concatenated hidden state and input
- output weight matrix
- zero-initialized biases
- `forward` method for one time step

The hidden state uses `tanh` activation, and the output uses `softmax`.

### Task 1: RNN Forward Propagation

Defines the `rnn` function that:

- iterates through all time steps
- stores every hidden state
- stores every output

Returns:

- `H`: all hidden states
- `Y`: all outputs

### Task 2: GRU Cell

Defines the `GRUCell` class with:

- update gate
- reset gate
- candidate hidden state
- output layer

The `forward` method computes the next hidden state and output for one
time step.

### Task 3: LSTM Cell

Defines the `LSTMCell` class with:

- forget gate
- update gate
- candidate cell state
- output gate
- hidden and cell state updates

The `forward` method returns:

- `h_next`: next hidden state
- `c_next`: next cell state
- `y`: output of the cell

### Task 4: Deep RNN

Defines the `deep_rnn` function that performs forward propagation for a
stack of recurrent layers.

It propagates:

- across time steps
- through each layer at every time step

Returns:

- `H`: all hidden states for all layers
- `Y`: all outputs

## Core Concepts

### Hidden State Update

For a simple RNN:

```text
h_next = tanh([h_prev, x_t]Wh + bh)
```

### Softmax Output

```text
y = softmax(h_nextWy + by)
```

### GRU Gating

GRUs use gates to control how much previous information is kept or
replaced.

### LSTM Memory

LSTMs maintain a separate cell state, which helps preserve information
across longer sequences.

### Deep Recurrent Models

Deep RNNs stack multiple recurrent layers so that higher layers can
learn more abstract sequence representations.

## Usage

Run the sample files from inside the `RNNs` directory:

```bash
python3 0-main.py
python3 1-main.py
python3 2-main.py
python3 3-main.py
python3 4-main.py
```

## Repository

- **Repository**: `holbertonschool-machine_learning`
- **Directory**: `supervised_learning/RNNs`
- **Language**: Python

---

**Author**: Arber Zylyftari
