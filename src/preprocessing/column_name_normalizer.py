import re
import pandas as pd
from .base import Preprocessor


class ColumnNameNormalizer(Preprocessor):
    """
    Standardize column names for consistency across datasets.

    Parameters
    ----------
    lowercase : bool, default True
        Convert column names to lowercase.
    strip : bool, default True
        Remove leading and trailing whitespace.
    replace_spaces : bool, default True
        Replace spaces in column names with underscores.
    valid_chars_pattern : str, default r"[^a-zA-Z0-9_]"
        Regular expression defining characters to remove.

    Returns
    -------
    DataFrame
        DataFrame with normalized column names.

    Notes
    -----
    This Preprocessor performs only renaming and does not alter data values.

    Operations include:
    - strip whitespace
    - lowercase
    - replace spaces with underscores
    - remove invalid characters
    """
    def __init__(
        self,
        lowercase: bool = True,
        strip: bool = True,
        replace_spaces: bool = True,
        valid_chars_pattern: str = r"[^a-zA-Z0-9_]",
    ):
        self.lowercase = lowercase
        self.strip = strip
        self.replace_spaces = replace_spaces
        self.valid_chars_pattern = valid_chars_pattern

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        new_cols = []

        for col in df.columns:
            name = col

            if self.strip:
                name = name.strip()

            if self.lowercase:
                name = name.lower()

            if self.replace_spaces:
                name = name.replace(" ", "_")

            # remove invalid chars
            name = re.sub(self.valid_chars_pattern, "", name)

            new_cols.append(name)

        df.columns = new_cols
        return df
