from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from scipy.special import hermite
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass

from cross_validation import QuantileStratifiedKFold


@dataclass
class FoldOptimizationResult:
    """Store optimization results for each fold"""
    optimal_mu: float
    rmse_values: np.ndarray
    mu_values: np.ndarray
    optimal_rmse: float
    weight: float = 0.0


class LinearNonlinearMixedRegressor(BaseEstimator, RegressorMixin):
    """
    A mixed regressor that combines linear and non-linear models using Hermite polynomials.

    Parameters
    ----------
    mu : str or float, default='auto'
        The non-linearity-propensity parameter.
        If 'auto', it will be estimated during fitting.
        If float, it will be used as a fixed value. Must be between 0 and 1.
    n : int, default=4
        The degree of Hermite polynomials to use.
    n_folds : int, default=5
        Number of folds for cross-validation when estimating mu.
    n_mu_points : int, default=100
        Number of mu values to test in grid search.
    random_state : int or None, default=None
        Random state for reproducibility.
    """

    def __init__(self,
                 mu: str | float = 'auto',
                 n: int = 4,
                 n_folds: int = 5,
                 n_mu_points: int = 100,
                 random_state: Optional[int] = None):
        self.mu = mu
        self.n = n
        self.n_folds = n_folds
        self.n_mu_points = n_mu_points
        self.random_state = random_state
        self.qskf = QuantileStratifiedKFold(n_splits=self.n_folds, random_state=self.random_state)
        self.y_mean = 0
        self.linear_model = LinearRegression(fit_intercept=False)
        self.nonlinear_model = LinearRegression(fit_intercept=False)
        self._mu_value = None
        self.fold_results_ = None

    def fit(self,
            X: np.ndarray,
            y: np.ndarray) -> 'LinearNonlinearMixedRegressor':
        """
        Fit the Linear Non Linear Mixed Model

        Parameters
        ----------
        X: np.ndarray
            Training data
        y: np: ndarray
            Target values

        Returns
        -------
        LinearNonlinearMixedRegressor
            Fitted estimator
        """
        if isinstance(self.mu, str) and self.mu == 'auto':
            self._mu_value = self._estimate_mu(X, y)
        else:
            self._mu_value = float(self.mu)
            self._estimate_y_mean(y)

        # Compute Hermite features
        X_hermite = self._compute_hermite_features(X)

        # Fit final models on full dataset
        self._fit_models(X, y, X_hermite)

        return self

    def predict(self,
                X: np.ndarray) -> np.ndarray:
        """
        Predict using the LNLM model

        Parameters
        ----------
        X: np.ndarray
            Samples

        Returns
        -------

        """

        X_hermite = self._compute_hermite_features(X)
        return (self.y_mean +
                self._mu_value * self.nonlinear_model.predict(X_hermite) +
                (1 - self._mu_value) * self.linear_model.predict(X))

    def _estimate_mu(self,
                     X: np.ndarray,
                     y: np.ndarray) -> float:
        """
        Estimate optimal mu using quantile k-fold cross-validation with weighted averaging.

        Parameters
        ----------
        X: np.ndarray
            Training data
        y: np.ndarray
            Target values

        Returns
        -------
        float
            Estimated mu
        """
        fold_results: List[FoldOptimizationResult] = []

        for fold_idx, (train_idx, val_idx) in enumerate(self.qskf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            X_hermite_train = self._compute_hermite_features(X_train)
            X_hermite_val = self._compute_hermite_features(X_val)

            # Fit models on training data
            self._estimate_y_mean(y_train)
            self._fit_models(X_train, y_train, X_hermite_train)

            # Optimize mu for this fold
            fold_result = self._optimize_fold(X_val, y_val, X_hermite_val)

            # Compute weight for this fold based on RMSE reduction
            xi = np.sqrt(np.mean((fold_result.rmse_values - fold_result.optimal_rmse) ** 2))
            fold_result.weight = xi

            fold_results.append(fold_result)

        self.fold_results_ = fold_results

        # Compute weighted average of optimal mu values
        total_weight = sum(result.weight for result in fold_results)
        if total_weight > 0:
            mu_star = sum(
                result.optimal_mu * result.weight
                for result in fold_results
            ) / total_weight
        else:
            mu_star = np.mean([result.optimal_mu for result in fold_results])

        return mu_star

    def _compute_hermite_features(self,
                                  X: np.ndarray) -> np.ndarray:
        """
        Compute Hermite polynomial features

        Parameters
        ----------
        X: np.ndarray
            Training samples

        Returns
        -------
        np.ndarray
            Computed Hermite polynomials
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        H = np.zeros((X.shape[0], X.shape[1] * self.n))

        for i in range(X.shape[1]):
            for j in range(self.n):
                poly = hermite(j)
                start_idx = i * self.n + j
                H[:, start_idx] = poly(X[:, i])

        return H

    def _compute_rmse(self,
                      mu: float,
                      X: np.ndarray,
                      y: np.ndarray,
                      X_hermite: np.ndarray) -> float:
        """
        Compute RMSE for a given mu value

        Parameters
        ----------
        mu: float
            The nonlinearity propensity parameter
        X: np.ndarray
            Training samples
        y: np.ndarray
            Target values
        X_hermite: np.ndarray
            Hermite polynomials

        Returns
        -------
        float
            RMSE
        """
        y_pred = (self.y_mean +
                  mu * self.nonlinear_model.predict(X_hermite) +
                  (1 - mu) * self.linear_model.predict(X))
        return root_mean_squared_error(y, y_pred)

    def _optimize_fold(self,
                       X_val: np.ndarray,
                       y_val: np.ndarray,
                       X_hermite_val: np.ndarray) -> FoldOptimizationResult:
        """
        Optimize mu for a single fold and compute all necessary metrics.

        Parameters
        ----------
        X_val: np.ndarray
            Validation samples
        y_val: np.ndarray
            Validation target values
        X_hermite_val: np.ndarray
            Validation Hermite polynomials

        Returns
        -------
        FoldOptimizationResult
            Results for this fold
        """
        # Create grid of mu values
        mu_values = np.linspace(1e-6, 1, self.n_mu_points)
        rmse_values = np.array([
            self._compute_rmse(mu, X_val, y_val, X_hermite_val)
            for mu in mu_values
        ])

        # Find optimal mu and its corresponding RMSE
        optimal_idx = np.argmin(rmse_values)
        optimal_mu = mu_values[optimal_idx]
        optimal_rmse = rmse_values[optimal_idx]

        return FoldOptimizationResult(
            optimal_mu=optimal_mu,
            rmse_values=rmse_values,
            mu_values=mu_values,
            optimal_rmse=optimal_rmse
        )

    def _fit_models(self,
                    X: np.ndarray,
                    y: np.ndarray,
                    X_hermite: np.ndarray) -> None:
        """Fit both linear and non-linear models."""
        self.nonlinear_model.fit(X_hermite, y)
        self.linear_model.fit(X, y)

    def _estimate_y_mean(self,
                         y: np.ndarray) -> None:
        """Estimate and store the mean of y."""
        self.y_mean = np.mean(y)
