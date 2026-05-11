# Clustering

Implementations of clustering algorithms including K-means, Gaussian Mixture Models (GMM) with Expectation-Maximization, and agglomerative clustering — all built from scratch using NumPy.

## Tasks

| File | Function | Description |
|------|----------|-------------|
| `0-initialize.py` | `initialize(X, k)` | Initializes K-means centroids using a multivariate uniform distribution |
| `1-kmeans.py` | `kmeans(X, k, iterations=1000)` | Performs K-means clustering; reinitializes empty clusters automatically |
| `2-variance.py` | `variance(X, C)` | Computes the total intra-cluster variance for a set of centroids |
| `3-optimum.py` | `optimum_k(X, kmin=1, kmax=None, iterations=1000)` | Finds the optimum number of clusters by testing values from kmin to kmax |
| `4-initialize.py` | `initialize(X, k)` | Initializes GMM parameters (priors, means, covariances) using K-means |
| `5-pdf.py` | `pdf(X, m, S)` | Computes the probability density function of a Gaussian distribution |
| `6-expectation.py` | `expectation(X, pi, m, S)` | E-step: computes posterior probabilities (responsibilities) for each cluster |
| `7-maximization.py` | `maximization(X, g)` | M-step: updates GMM parameters from responsibilities |
| `8-EM.py` | `expectation_maximization(X, k, ...)` | Runs the full EM algorithm until convergence |
| `9-BIC.py` | `BIC(X, kmin, kmax, ...)` | Selects the best number of GMM clusters using the Bayesian Information Criterion |
| `10-kmeans.py` | — | K-means clustering using `sklearn.cluster.KMeans` |
| `11-gmm.py` | — | Gaussian Mixture Model using `sklearn.mixture.GaussianMixture` |
| `12-agglomerative.py` | — | Agglomerative hierarchical clustering with Ward linkage and dendrogram visualization |

## Key Concepts

- **K-means**: Iteratively assigns points to the nearest centroid and recomputes centroids until convergence
- **GMM / EM**: Models data as a mixture of Gaussians; EM alternates between computing responsibilities (E-step) and updating parameters (M-step)
- **BIC**: Penalizes model complexity — lower BIC indicates a better balance between fit and number of parameters

## Requirements

- Python 3.x
- NumPy
- scikit-learn
- scipy
- Matplotlib
