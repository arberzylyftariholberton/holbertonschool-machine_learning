# Calculus

Implementations of core calculus concepts applied to sequences and polynomials.

## Tasks

| File | Function | Description |
|------|----------|-------------|
| `9-sum_total.py` | `summation_i_squared(n)` | Calculates the sum of the series ∑i² from i=1 to n using the closed-form formula `n(n+1)(2n+1)/6` |
| `10-matisse.py` | `poly_derivative(poly)` | Returns the derivative of a polynomial represented as a list of coefficients |
| `17-integrate.py` | `poly_integral(poly, C=0)` | Returns the integral of a polynomial represented as a list of coefficients, with optional constant C |

## Polynomial Representation

Polynomials are represented as lists where the index corresponds to the power of x:

```python
# f(x) = 5 + 3x + x^2  →  [5, 3, 1]
poly_derivative([5, 3, 1])   # → [3, 2]
poly_integral([5, 3, 1])     # → [0, 5, 1.5, 1/3]
```

## Requirements

- Python 3.x
