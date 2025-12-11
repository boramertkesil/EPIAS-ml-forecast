"""
Load dataset for EPİAŞ UECM Data
------------------------------------------------

This module loads the raw EPİAŞ UECM (Uzlaştırmaya Esas Çekiş Miktarı)
files from ``data/raw/``.

Input Data Requirements
-----------------------
Each raw CSV must include:

    - ``Date`` : string in DD.MM.YYYY format  
    - ``Hour`` : string in HH:MM format  
    - ``MWh``  : numeric value using EU formatting, e.g. '27.138,70'
"""
import pandas as pd
from pathlib import Path
import re

def load_raw_data(path: str = 'data/raw') -> pd.DataFrame:
    """
    Load and merge all CSV files matching 'epias_UECM_*.csv' in given directory.

    Parameters
    ----------
    path : str, optional
        Directory containing the raw UECM CSV files.

    Returns
    -------
    DataFrame
        Concatenated CSV files.
    """
    directory = Path(path)

    pattern = re.compile(r'^epias_UECM_.*\.csv$')
    csv_files = [f for f in directory.iterdir() if pattern.match(f.name)]

    if not csv_files:
        raise FileNotFoundError("No files matching 'epias_UECM_*.csv' were found.")

    dfs = []
    for file in csv_files:
        df = pd.read_csv(
            file,
            sep=';',
            thousands='.',
            decimal=',',
            dtype={'Date': 'string', 'Hour': 'string', 'MWh': 'float64'},
        )
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)