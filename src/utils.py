# utils.py

"""
My tools for data analysis
"""

from pathlib import Path
from typing import Annotated, Dict, Tuple, Callable, Any, List
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def pipe(
    raw_input: Any,
    *functions,
    **functions_with_args
) -> Any:
    """
    Creates a pipeline (or chain) for every function. Basically,
    this function initially accepts a data then passes it to the next
    function, then the output passes it to the next function as input.

    Args:
        raw_input (Any): Any input, could be list, tuple, etc.

    Other Parameters:
        param1 (Callable): Any function with only one argument.
        param2 (Callable): Any function with only one argument.
        ...

    Keyword Args:
        key1 (Callable): Any function with one or more than one
            arguments with arguments written as list.
        key2 (Callable): Any function with one or more than one
            arguments with arguments written as list.
        ...

    Returns:
        Any: Any output as a result of the functions it goes through.
    """

    # TODO: Needs more improvement for robustness.
    # Currently it will only work for some cases.
    output = raw_input

    if functions:
        for function in functions:
            output = function(output)

    if functions_with_args:
        for function, args_list in functions_with_args.items():
            output = eval(function)(output, *args_list)

    return output


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


def get_unique_df(df: pd.DataFrame, return_counts: bool = False) -> np.ndarray:
    """Get unique values and their distribution from the whole DataFrame.

    Args:
        df (pd.DataFrame): The whole DataFrame input.
        return_counts (bool): An option to get the distribution of
            the unique values or not.

    Returns:
        np.ndarray: An array that shows the unique values from the
            DataFrame and their distribution if opted.
    """
    return np.array(
        [
            np.unique(
                df.loc[:, col],
                return_counts=return_counts
            )
            for col in df
        ]
    )


def plot_crosstab(cross: pd.DataFrame):
    data = [
        go.Bar(
            name=str(x),
            x=cross.index,
            y=cross[x]
        ) for x in cross.columns
    ]

    fig = go.Figure(data)
    fig.update_layout(barmode='stack')
    return fig


def word_tokenize(text: str) -> List[str]:
    """Converts a text (in string format) to a list of string.

    Args:
        text (str): Raw text format, usually having a '.txt' file extension.

    Returns:
        List[str]: List of strings from the raw text.
    """

    first_pattern = r'[A-Za-z]{2,}'
    second_pattern = r'W+^[\s+]'
    new_text = re.sub(second_pattern, '', text)
    return re.findall(first_pattern, new_text)


def get_dataPath(filename: str, directory: str = "raw") -> Path:
    data_dir = Path.joinpath(Path.cwd().parent, "data/")
    return Path.joinpath(data_dir, directory, filename)


def get_data(filename: str, directory: str = "raw") -> str:
    file = get_dataPath(filename, directory)
    with open(file, encoding="utf-8") as f:
        data = f.read()
    return data


class AnnotationFactory:
    """Copied from RealPython

    Link: https://realpython.com/python39-new-features/
    """

    def __init__(self, type_hint):
        self.type_hint = type_hint

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return Annotated[(self.type_hint,) + key]
        else:
            return Annotated[self.type_hint, key]

    def __repr__(self):
        return f"{self.__class__.__name__}({self.type_hint})"
