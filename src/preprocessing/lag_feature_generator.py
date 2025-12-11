import pandas as pd
from .base import Preprocessor

class LagFeatureGenerator(Preprocessor):
    """
    Generate lagged features for selected columns.

    Parameters
    ----------
    columns : list of str
        Columns for which to create lag features.
    lags : list of int, default [1, 24, 168, 336]
        Lag offsets (in number of rows) to apply.
        For hourly data:
            1   = previous hour
            24  = same hour previous day
            168 = same hour previous week
            336 = same hour two weeks ago
    Notes
    -----
    Lag features naturally introduce missing values at the beginning of
    the series (e.g., lag_24 creates 24 initial NaNs). This is expected
    and correct behavior. Such NaN values should and can be handled by
    a MissingValueImputer.
    """
    def __init__(self, columns, lags=None):
        self.columns = columns
        self.lags = lags if lags is not None else [1, 24, 168, 336]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Validate inputs
        for col in self.columns:
            if col not in df.columns:
                raise ValueError(f"LagFeatureGenerator: column '{col}' not found in DataFrame.")

        for lag in self.lags:
            if not isinstance(lag, int) or lag <= 0:
                raise ValueError("Lags must be positive integers.")

        # Create lag features
        for col in self.columns:
            for lag in self.lags:
                new_col = f"{col}_lag_{lag}"
                df[new_col] = df[col].shift(lag)

        return df
