import pandas as pd
from .base import Preprocessor

class DateTimeIndexer(Preprocessor):
    """
    Combine separate date and time columns into a DatetimeIndex.

    Parameters
    ----------
    date_col : str
        Column containing date values.
    time_col : str
        Column containing time values.
    date_format : str, default "%d.%m.%Y"
        Format string for parsing date values.
    time_format : str, default "%H:%M"
        Format string for parsing time values.
    drop_original : bool, default True
        If True, remove date and time columns after indexing.
    errors : {"raise", "coerce"}, default "raise"
        Error handling strategy for invalid datetime parsing.

    Returns
    -------
    DataFrame
        DataFrame indexed by the parsed datetime values.

    Notes
    -----
    The resulting index is sorted in ascending order.
    """
    def __init__(
        self,
        date_col: str,
        time_col: str,
        date_format: str = "%d.%m.%Y",
        time_format: str = "%H:%M",
        drop_original: bool = True,
        errors: str = "raise",
    ):
        self.date_col = date_col
        self.time_col = time_col
        self.date_format = date_format
        self.time_format = time_format
        self.drop_original = drop_original
        self.errors = errors

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Required column checks
        if self.date_col not in df.columns:
            raise ValueError(f"Missing required column '{self.date_col}'.")

        if self.time_col not in df.columns:
            raise ValueError(f"Missing required column '{self.time_col}'.")

        # Merge into a single datetime string
        datetime_str = df[self.date_col].astype(str) + " " + df[self.time_col].astype(str)

        # Build combined format
        datetime_format = f"{self.date_format} {self.time_format}"

        # Parse datetime
        df["datetime"] = pd.to_datetime(
            datetime_str,
            format=datetime_format,
            errors=self.errors,
        )

        # Set index
        df = df.set_index("datetime")

        # Drop original columns if configured
        if self.drop_original:
            df = df.drop(columns=[self.date_col, self.time_col])

        # Sort index to maintain temporal consistency
        df = df.sort_index()

        return df