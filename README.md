# Polymodels

This repository contains a Python implementation of polymodels. Polymodels are a collection of non-linear univariate models introduced in:

> Barrau, T., & Douady, R. (2022). Artificial Intelligence for Financial Markets: The Polymodel Approach. Springer.

## Overview
A polymodel can be defined as a collection of models, all equally valid and significant, that can be understood as a collection of relevant points of view on the same reality. Mathematically, it can be equally formalized as:

$$\begin{align} Y &= \phi_1(X_1) \\
Y &= \phi_2(X_2) \\ 
&... \\ 
Y &= \phi_n(X_n) 
\end{align}$$

Where $Y$ is the target variable, $X_i$ and $\phi_i$ are respectively the explanatory variable and the function of the $i^{th}$ model, with $i \in [1, n]$, and $n$ is the number of models (and factors).

Each model $\phi_i$ is a univariate model called a Linear Non Linear Mixed Model (LNLM), representing a target variable $Y$ as follows:

$$LNLM(X) = \bar{y} + \mu\sum_{h=1}^{u}\hat{\beta}_h^{NonLin}H_h(X) + (1-\mu)\hat{\beta}^{Lin}X + \epsilon$$

Where:
* $H_h(X)$ is a Hermite polynomial defined as $H_h(x) = (-1)^he^\frac{x^2}{2}\frac{d^h}{dx^h}e^{\frac{-x^2}{2}}$
* $\mu$ is the non-linearity propensity parameter determined through cross-validation
* $\bar{y}$ is the mean of the target variable

## Key Features
- **reduced overfitting**: since it's a set of univariate regression, it reduces overfitting (less subject to curse of dimensionality and multicollinearity) 
- **increasing precision**: thanks to Hermite polynomials, you can model nonlinear relationships, which alleviates underfitting compared to OLS.
- **increasing robustness**:  due to the disappearance of multicollinearity, parameter instability problem is improved. Regarding missing data, since each variable is fitted independently, using its own elementary model, there is no need to remove any observations, the number of observations used in each elementary model can be different.

## Installation
[TODO]
```bash
pip install polymodels
```

## Basic Usage

```python
from polymodels.lnlm import LNLM

# Initialize the model
model = LNLM(n_folds=5)  # 5-fold cross validation

# Fit the model
model.fit(X, y)

# Make predictions
y_pred = model.predict(X_new)
```

## Requirements

- numpy
- scikit-learn
- pandas

## Disclaimer
⚠️ This is an unofficial implementation of the LNLM model. I am not affiliated with the authors of the original paper, and this implementation is provided as-is without any guarantees. Use at your own risk.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
