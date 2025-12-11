import pandas as pd
import holidays
from .base import Preprocessor

class HolidayFeatureExtractor(Preprocessor):
    """
    Add holiday-based features using a country-specific holiday calendar.

    Parameters
    ----------
    country : str, default "TR"
        Country code for the holidays library.
    include_pre_post : bool, default False
        Add is_pre_holiday and is_post_holiday flags.
    include_name : bool, default False
        Add holiday_name column.
    """
    def __init__(
        self,
        country: str = "TR",
    ):
        self.country = country
        self.hcal = None

    def _load_calendar(self, years):
        try:
            return holidays.country_holidays(self.country, years=years)
        except Exception as e:
            raise ValueError(
                f"Failed to load holiday calendar for country='{self.country}' "
                f"and years={years}. Original error: {e}"
            )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("HolidayFeatureExtractor requires a DatetimeIndex.")

        df = df.copy()

        # Dynamically load holiday calendar for relevant years
        years = df.index.year.unique().tolist()
        self.hcal = self._load_calendar(years)

        # Normalize and convert to date
        dates = df.index.tz_localize(None).normalize().date

        # Holiday boolean flag
        df["is_holiday"] = [d in self.hcal for d in dates]
        df["is_holiday"] = df["is_holiday"].astype(int)

        return df
