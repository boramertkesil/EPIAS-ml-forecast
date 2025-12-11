import pandas as pd
from .base import Preprocessor

class TimeFeatureExtractor(Preprocessor):
    """
    Extract time-based features from a DatetimeIndex.

    Parameters
    ----------
    features : list of {"hour", "day_of_week", "day_of_month", "day_of_year",
                        "month", "week_of_year", "is_weekend",
                        "is_month_start", "is_month_end",
                        "is_quarter_start", "is_quarter_end"},
               optional
        Time features to compute. If None, defaults to common features.
    """
    def __init__(self, features=None):
        self.features = features or [
            "hour",
            "day_of_week",
            "month",
            "is_weekend",
        ]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("TimeFeatureExtractor requires a DatetimeIndex.")

        df = df.copy()

        dt_index: pd.DatetimeIndex = df.index

        if "hour" in self.features:
            df["hour"] = dt_index.hour

        if "day_of_week" in self.features:
            df["day_of_week"] = dt_index.dayofweek

        if "day_of_month" in self.features:
            df["day_of_month"] = dt_index.day

        if "day_of_year" in self.features:
            df["day_of_year"] = dt_index.dayofyear

        if "month" in self.features:
            df["month"] = dt_index.month

        if "week_of_year" in self.features:
            df["week_of_year"] = dt_index.isocalendar().week.astype(int)

        if "is_weekend" in self.features:
            df["is_weekend"] = (dt_index.dayofweek >= 5).astype(int)

        return df
