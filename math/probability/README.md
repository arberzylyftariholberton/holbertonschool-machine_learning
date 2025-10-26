# Probability Distributions

This project implements various probability distributions from scratch in Python, including Poisson, Exponential, Normal, and Binomial distributions.

## Table of Contents

- [Poisson Distribution](#poisson-distribution)
- [Exponential Distribution](#exponential-distribution)
- [Normal Distribution](#normal-distribution)
- [Binomial Distribution](#binomial-distribution)

---

## Poisson Distribution

The Poisson distribution models the number of events occurring in a fixed interval of time or space.

### Class: `Poisson`

**Constructor:**
```python
Poisson(data=None, lambtha=1.)
```

**Parameters:**
- `data` (list, optional): Data to estimate the distribution
- `lambtha` (float): Expected number of occurrences (λ > 0)

**Attributes:**
- `lambtha` (float): The rate parameter λ

**Methods:**

#### `pmf(k)`
Calculates the Probability Mass Function for a given number of successes.

- **Parameters:** `k` (int) - Number of successes
- **Returns:** `float` - PMF value for k
- **Formula:** `P(X = k) = (e^(-λ) × λ^k) / k!`

#### `cdf(k)`
Calculates the Cumulative Distribution Function for a given number of successes.

- **Parameters:** `k` (int) - Number of successes
- **Returns:** `float` - CDF value for k
- **Formula:** `P(X ≤ k) = Σ(i=0 to k) [e^(-λ) × λ^i / i!]`

**Real-World Applications:**
- Number of customer arrivals per hour
- Number of emails received per day
- Number of defects in manufacturing
- Call center call volume

---

## Exponential Distribution

The Exponential distribution models the time between events in a Poisson process.

### Class: `Exponential`

**Constructor:**
```python
Exponential(data=None, lambtha=1.)
```

**Parameters:**
- `data` (list, optional): Data to estimate the distribution
- `lambtha` (float): Expected number of occurrences in a time period (λ > 0)

**Attributes:**
- `lambtha` (float): The rate parameter λ

**Methods:**

#### `pdf(x)`
Calculates the Probability Density Function for a given time period.

- **Parameters:** `x` (float) - Time period
- **Returns:** `float` - PDF value for x
- **Formula:** `f(x) = λ × e^(-λx)` for x ≥ 0

#### `cdf(x)`
Calculates the Cumulative Distribution Function for a given time period.

- **Parameters:** `x` (float) - Time period
- **Returns:** `float` - CDF value for x
- **Formula:** `P(X ≤ x) = 1 - e^(-λx)` for x ≥ 0

**Real-World Applications:**
- Time until next customer arrival
- Lifespan of electronic components
- Time between earthquakes
- Service time in queuing systems

---

## Normal Distribution

The Normal (Gaussian) distribution is the most common continuous probability distribution.

### Class: `Normal`

**Constructor:**
```python
Normal(data=None, mean=0., stddev=1.)
```

**Parameters:**
- `data` (list, optional): Data to estimate the distribution
- `mean` (float): Mean of the distribution (μ)
- `stddev` (float): Standard deviation (σ > 0)

**Attributes:**
- `mean` (float): The mean μ
- `stddev` (float): The standard deviation σ

**Methods:**

#### `z_score(x)`
Calculates the z-score of a given x-value.

- **Parameters:** `x` (float) - The x-value
- **Returns:** `float` - The z-score
- **Formula:** `z = (x - μ) / σ`

#### `x_value(z)`
Calculates the x-value of a given z-score.

- **Parameters:** `z` (float) - The z-score
- **Returns:** `float` - The x-value
- **Formula:** `x = μ + z × σ`

#### `pdf(x)`
Calculates the Probability Density Function for a given x-value.

- **Parameters:** `x` (float) - The x-value
- **Returns:** `float` - PDF value for x
- **Formula:** `f(x) = (1 / (σ√(2π))) × e^(-(x-μ)²/(2σ²))`

#### `cdf(x)`
Calculates the Cumulative Distribution Function for a given x-value.

- **Parameters:** `x` (float) - The x-value
- **Returns:** `float` - CDF value for x
- **Formula:** `P(X ≤ x) = 0.5 × [1 + erf((x - μ) / (σ√2))]`

---

## Binomial Distribution

The Binomial distribution models the number of successes in a fixed number of independent trials.

### Class: `Binomial`

**Constructor:**
```python
Binomial(data=None, n=1, p=0.5)
```

**Parameters:**
- `data` (list, optional): Data to estimate the distribution
- `n` (int): Number of Bernoulli trials
- `p` (float): Probability of success (0 < p < 1)

**Attributes:**
- `n` (int): Number of trials
- `p` (float): Probability of success

**Parameter Estimation from Data:**

When data is provided, the class estimates n and p using:
1. Calculate initial p: `p = 1 - (variance / mean)`
2. Calculate n: `n = mean / p`
3. Round n to nearest integer
4. Recalculate p: `p = mean / n`

**Methods:**

#### `pmf(k)`
Calculates the Probability Mass Function for a given number of successes.

- **Parameters:** `k` (int) - Number of successes
- **Returns:** `float` - PMF value for k
- **Formula:** `P(X = k) = C(n,k) × p^k × (1-p)^(n-k)` where `C(n,k) = n! / (k! × (n-k)!)`

#### `cdf(k)`
Calculates the Cumulative Distribution Function for a given number of successes.

- **Parameters:** `k` (int) - Number of successes
- **Returns:** `float` - CDF value for k
- **Formula:** `P(X ≤ k) = Σ(i=0 to k) [C(n,i) × p^i × (1-p)^(n-i)]`

---

## Mathematical Background

### Discrete vs Continuous Distributions

**Discrete Distributions** (Poisson, Binomial):
- Deal with countable outcomes (0, 1, 2, 3, ...)
- Use PMF (Probability Mass Function)
- P(X = k) gives exact probability

**Continuous Distributions** (Normal, Exponential):
- Deal with continuous values (any real number)
- Use PDF (Probability Density Function)
- P(X = x) = 0; use intervals instead
- PDF values can exceed 1 (it's a density, not probability)

### Mean and Variance

| Distribution | Mean | Variance |
|--------------|------|----------|
| Poisson | λ | λ |
| Exponential | 1/λ | 1/λ² |
| Normal | μ | σ² |
| Binomial | n×p | n×p×(1-p) |

### Probability Rules

**Complement Rule:**
```
P(X > k) = 1 - P(X ≤ k) = 1 - cdf(k)
```

**Interval Probability (Continuous):**
```
P(a < X < b) = cdf(b) - cdf(a)
```

**Interval Probability (Discrete):**
```
P(a ≤ X ≤ b) = cdf(b) - cdf(a-1)
```

---

## Error Handling

All classes include comprehensive input validation:

**ValueError:**
- `"lambtha must be a positive value"`
- `"stddev must be a positive value"`
- `"p must be greater than 0 and less than 1"`
- `"n must be a positive value"`
- `"data must contain multiple values"`

**TypeError:**
- `"data must be a list"`

---

## Requirements

- Python 3.x
- No external libraries required for core functionality

## File Structure
```
probability/
├── poisson.py      # Poisson distribution class
├── exponential.py  # Exponential distribution class
├── normal.py       # Normal distribution class
├── binomial.py     # Binomial distribution class
└── README.md       # This file
```
