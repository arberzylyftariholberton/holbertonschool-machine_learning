# Dimensionality Reduction

Principal Component Analysis (PCA) implemented from scratch using NumPy via Singular Value Decomposition (SVD).

## Tasks

| File | Function | Description |
|------|----------|-------------|
| `0-pca.py` | `pca(X, var=0.95)` | Returns the weight matrix W that preserves a given fraction of the original variance. Selects the minimum number of principal components needed to retain `var` fraction of variance. |
| `1-pca.py` | `pca(X, ndim)` | Projects dataset X onto `ndim` principal components. Mean-centers the data before decomposition and returns the transformed matrix T of shape `(n, ndim)`. |

## How It Works

Both implementations use `np.linalg.svd` to decompose the data matrix:

```python
U, S, Vh = np.linalg.svd(X)
W = Vh.T[:, :nd]    # weight matrix: (d, nd)
T = X_centered @ W  # transformed data: (n, nd)
```

- `0-pca.py` selects `nd` automatically based on the cumulative explained variance threshold
- `1-pca.py` accepts a fixed target dimensionality `ndim`

## Requirements

- Python 3.x
- NumPy
