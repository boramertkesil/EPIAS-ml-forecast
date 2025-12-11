from abc import ABC, abstractmethod
import pandas as pd

class Preprocessor(ABC):
    """Base class for all Preprocessors."""

    def fit(self, df: pd.DataFrame) -> 'Preprocessor':
        # Default: stateless transformer
        return self

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply transformation to df and return a new df."""
        pass

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)