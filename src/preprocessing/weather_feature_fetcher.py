import pandas as pd
from .base import Preprocessor

class WeatherFeatureFetcher(Preprocessor):
    """
    Fetch hourly historical weather features (currently temperature) from the
    Open-Meteo Archive API and merge them into the dataset.

    This preprocessor is intentionally designed to be extensible. In future
    versions, additional weather variables such as wind speed, humidity,
    solar radiation, cloud cover, precipitation, etc., can be fetched
    simply by adding them to the 'hourly' parameter list and expanding
    the returned feature set.

    Parameters
    ----------
    latitude : float
        Geographic latitude of the target location.
        Determines where weather data should be fetched from.
        For example, Istanbul ≈ 41.0082.

    longitude : float
        Geographic longitude of the target location.
        Determines the location of the weather station grid.
        For example, Istanbul ≈ 28.9784.

    temp_col : str, default "temperature"
        The name of the column under which temperature will be stored
        after merging with the dataset.
    """

    def __init__(self, latitude, longitude, temp_col="temperature"):
        self.latitude = latitude
        self.longitude = longitude
        self.temp_col = temp_col

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Import inside method with clear error if missing.
        try:
            import requests
        except ImportError:
            raise ImportError(
                "WeatherFeatureFetcher requires the 'requests' library."
                "You can safely remove this step from preprocessing steps if not used."
            )

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("WeatherFeatureFetcher requires a DatetimeIndex.")

        df = df.copy()

        # Use exact start and end timestamps.
        start_date = df.index.min().strftime("%Y-%m-%d")
        end_date   = df.index.max().strftime("%Y-%m-%d")

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "hourly": ["temperature_2m"],
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "UTC",
        }

        response = requests.get(url, params=params)
        data = response.json()

        if "hourly" not in data or "time" not in data["hourly"]:
            raise ValueError(f"Open-Meteo returned an unexpected response: {data}")

        hourly = data["hourly"]
        weather = pd.DataFrame({
            "datetime": pd.to_datetime(hourly["time"]),
            self.temp_col: hourly["temperature_2m"],
        }).set_index("datetime")

        return df.merge(weather, how="left", left_index=True, right_index=True)
