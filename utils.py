# utils.py

"""
My tools for data analysis
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def get_freq(series: pd.Series) -> np.ndarray:
    """Get frequency distribution for a given series.

    Args:
        series (pd.Series): A column (series) in a DataFrame.

    Returns:
        np.ndarray: An array of frequencies for each distinct key.
    """
    return series.value_counts().values


def get_perc(series: pd.Series) -> np.ndarray:
    """Get percentage distribution for a given series.

    Args:
        series (pd.Series): A column (series) in a DataFrame.

    Returns:
        np.ndarray: An array of percentages for each distinct key.
    """
    return series.value_counts(normalize=True).values.round(4)


def get_freq_perc(series: pd.Series) -> Tuple[int, float]:
    """Get frequency and percentage distribution for a given series.

    Args:
        series (pd.Series): A column (series) in a DataFrame.

    Returns:
        tuple: A tuple containing frequency and percentage, respectively.
    """
    return tuple(zip(get_freq(series), get_perc(series)))


def get_dist(series: pd.Series) -> Dict[int, Tuple[int, float]]:
    """Get distribution for a given series.

    Args:
        series (pd.Series): A column (series) in a DataFrame.

    Returns:
        dict: A dictionary in which the key is the distinct value of
        the data and the values of the dictionary is a tuple
        consisting of both frequency and percentage, respecdtively.
    """
    return dict(
        zip(
            series.value_counts().keys().values,
            get_freq_perc(series)
        )
    )


def get_dist_all(df: pd.DataFrame) -> dict:
    """Get distribution for all columns in a DataFrame.

    Args:
        df (pd.DataFrame): The whole DataFrame, if possible.

    Returns:
        dict: A dictionary for all columns, same with get_dist() function,
            the difference lies only in that in here, it takes all
            columns as input.
    """
    return {col: get_dist(df[col]) for col in df}
