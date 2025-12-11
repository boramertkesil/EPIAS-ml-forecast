import pandas as pd
from .base import Preprocessor

class MissingValueImputer(Preprocessor):
    """
    Fill missing values using a specified criterion.

    Parameters
    ----------
    criterion : {"mean", "median", "value", "ffill", "bfill"}, default "mean"
        Imputation method. "value" uses `fill_value` directly.
    columns : list of str, optional
        Columns to impute. If None, all numeric columns are used.
    fill_value : scalar, optional
        Constant value used when criterion="value".
    """
    def __init__(
        self,
        criterion: str = "mean",
        columns: list[str] | None = None,
        fill_value=None,
    ):
        self.criterion = criterion
        self.columns = columns
        self.fill_value = fill_value
        self.stats = {}

    def fit(self, df: pd.DataFrame):
        cols = self.columns or df.select_dtypes(include="number").columns

        if self.criterion == "mean":
            self.stats = df[cols].mean().to_dict()

        elif self.criterion == "median":
            self.stats = df[cols].median().to_dict()

        # "value", "ffill", "bfill" need no fitting
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        cols = self.columns or df.columns

        if self.criterion in {"mean", "median"}:
            df[cols] = df[cols].fillna(self.stats)

        elif self.criterion == "value":
            df[cols] = df[cols].fillna(self.fill_value)

        elif self.criterion == "ffill":
            df[cols] = df[cols].fillna(method="ffill")

        elif self.criterion == "bfill":
            df[cols] = df[cols].fillna(method="bfill")

        else:
            raise ValueError(f"Invalid criterion: {self.criterion}")

        return df