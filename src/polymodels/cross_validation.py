import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from typing import Iterator, Tuple


class QuantileStratifiedKFold(BaseEstimator):
    """
    Quantile-based stratified K-fold cross validator.

    This implementation splits the data into quantile buckets first, then assigns fold
    identifiers randomly within each quantile to ensure representative distribution
    across folds.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    shuffle : bool, default=True
        Whether to shuffle the data within quantiles.
    random_state : int or None, default=None
        Random seed for reproducibility.
    """

    def __init__(
            self,
            n_splits: int = 5,
            shuffle: bool = True,
            random_state: int = None
    ):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(
            self,
            X: np.ndarray,  # noqa
            y: np.ndarray
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices to split data into training and validation sets.

        Parameters
        ----------
        X : np.ndarray
            Training data, where n_samples is the number of samples.
        y : np.ndarray
            Target variable for stratification.

        Yields
        ------
        train_idx : ndarray
            Training set indices for current split.
        val_idx : ndarray
            Validation set indices for current split.
        """
        if self.n_splits < 2:
            raise ValueError(f"n_splits must be ≥ 2. Got {self.n_splits}")

        n_samples = len(y)
        indices = np.arange(n_samples)

        # Calculate number of quantile buckets
        n_quantiles = int(np.ceil(n_samples / self.n_splits))

        # Get quantile assignments for each sample
        y_series = pd.Series(y)
        quantile_labels = pd.qcut(y_series, n_quantiles, labels=False)

        # Initialize fold assignments array
        fold_assignments = np.zeros(n_samples, dtype=int)

        rng = np.random.RandomState(self.random_state)

        # Assign folds within each quantile
        for quantile in range(n_quantiles):
            quantile_mask = (quantile_labels == quantile)
            quantile_indices = indices[quantile_mask]
            n_samples_in_quantile = len(quantile_indices)

            # Generate fold assignments for this quantile
            if self.shuffle:
                # If number of samples in quantile is less than n_splits,
                # use only necessary number of fold identifiers
                n_folds_for_quantile = min(n_samples_in_quantile, self.n_splits)
                fold_ids = np.arange(n_folds_for_quantile)

                # Repeat fold ids if necessary to match quantile size
                n_repeats = int(np.ceil(n_samples_in_quantile / n_folds_for_quantile))
                fold_ids = np.tile(fold_ids, n_repeats)[:n_samples_in_quantile]

                # Shuffle the fold assignments
                rng.shuffle(fold_ids)
            else:
                # Without shuffling, assign folds sequentially
                fold_ids = np.arange(n_samples_in_quantile) % self.n_splits

            # Assign folds to samples in this quantile
            fold_assignments[quantile_mask] = fold_ids

        # Generate train/validation splits
        for fold_id in range(self.n_splits):
            # Get validation indices for current fold
            val_mask = (fold_assignments == fold_id)
            val_indices = indices[val_mask]

            # Get training indices (all other folds)
            train_indices = indices[~val_mask]

            yield train_indices, val_indices

    def get_n_splits(self) -> int:
        """Returns the number of splitting iterations in the cross-validator"""
        return self.n_splits
