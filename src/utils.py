# utils.py

"""
My tools for data analysis
"""

from pathlib import Path
from typing import Annotated, Dict, Tuple, Callable, Any, List, Union
import re
import inspect

import numpy as np
import pandas as pd
import plotly.graph_objects as go


class Utils:

    filename: str
    folder: str
    sheet_name: Union[str, int]

    def __init__(self, filename, folder, sheet_name=0) -> None:
        self.df = self.get_data(filename, folder, sheet_name)

    def get_path(self, filename: str, folder: str = "") -> Path:
        data_path = Path.joinpath(Path.cwd().parent, "data", folder)
        return Path.joinpath(data_path, filename)

    def get_data(
        self,
        filename: str,
        folder: str = "",
        sheet_name: Union[str, int] = 0,
        index_col: int = 0,
    ) -> pd.DataFrame:
        whole_path = self.get_path(filename, folder)
        try:
            return pd.read_csv(whole_path, index_col=index_col)
        except UnicodeDecodeError:
            return pd.read_excel(whole_path, sheet_name=sheet_name, index_col=index_col)

    def get_indices(
        self, start: str, end: Optional[str] = None
    ) -> Union[list[str], str]:

        start = self.df.index.get_loc(start)

        if end:
            end = self.df.index.get_loc(end)
            return self.df.index[start : end + 1].tolist()

        return self.df.index[start]

    def _get_indices(
        self, indices: Union[str, Tuple[str, str]]
    ) -> Union[str, List[str]]:
        """Helper function for the get_indices method."""

        try:
            indices = self.get_indices(indices)

        except KeyError:
            start, *_, end = indices
            indices = utils.get_indices(start, end)

        return indices

    def get_average(self, data: Union[pd.Series, pd.DataFrame, int, float]) -> float:
        """
        Robust average function of getting the average. Can handle different types
        of input.

        Args:
            data (Union[pd.Series, pd.DataFrame, int, float]): The data to be averaged.

        Returns:
            float: A single average (mean) value.
        """

        try:
            return data.apply(np.mean, axis=1).mean()
        except Exception:
            if isinstance(data, int) or isinstance(data, float):
                return data
            return data.mean()

    def collapse(
        self,
        indices_from: Union[str, Tuple[str, str]],
        indices_to: Union[str, Tuple[str, str]],
    ) -> float:
        """
        Collapse multiple columns of a dataframe into a single value by getting the mean.

        Args:
            indices_from (Union[str, Tuple[str, str]]): Column name or index name of a dataframe.
            indices_to (Union[str, Tuple[str, str]]): Column name or index name of a dataframe.

        Returns:
            float: Mean of all the values.
        """

        indices_from = self._get_indices_helper(indices_from)
        indices_to = self._get_indices_helper(indices_to)

        return self.get_average(self.df[indices_from].loc[indices_to])


def pipe(raw_input: Any, *functions, **functions_with_args) -> Any:
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
    return dict(zip(series.value_counts().keys().values, get_freq_perc(series)))


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
        [np.unique(df.loc[:, col], return_counts=return_counts) for col in df]
    )


def plot_crosstab(cross: pd.DataFrame):
    data = [go.Bar(name=str(x), x=cross.index, y=cross[x]) for x in cross.columns]

    fig = go.Figure(data)
    fig.update_layout(barmode="stack")
    return fig


def word_tokenize(text: str) -> List[str]:
    """Converts a text (in string format) to a list of string.

    Args:
        text (str): Raw text format, usually having a '.txt' file extension.

    Returns:
        List[str]: List of strings from the raw text.
    """

    first_pattern = r"[A-Za-z]{2,}"
    second_pattern = r"W+^[\s+]"
    new_text = re.sub(second_pattern, "", text)
    return re.findall(first_pattern, new_text)


def get_data_path(filename: str, directory: str = "raw") -> Path:
    data_dir = Path.joinpath(Path.cwd().parent, "data/")
    return Path.joinpath(data_dir, directory, filename)


def get_data(filename: str, directory: str = "raw") -> str:
    file = get_data_path(filename, directory)
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


def list_dir(path: Optional[Union[str, Path]] = None) -> List[str]:
    """List files and directories within the current directory.

    Args:
        path (Optional[Union[str, Path]]): Target path.

    Returns:
        List[str]: List of all files in the current directory.
    """


    if path is None:
        path = Path.cwd()

    return [file.name for file in Path(path).glob('**/*')]


def go_back(current_path: Path) -> Path:
    return current_path.parent


def go_source() -> Path:
    """
    Get path for top-level directory of project. Only
    works when git was initialized within the project.

    Args:
        None

    Returns:
        Path: Absolute path of top-level directory within
            the project.
    """

    path = Path.cwd()
    while True:
        if ".git" in list_dir(path):
            return path
        path = go_back(path)
    return path


def get_path(path: Optional[Path] = None, *args) -> Path:

    if path is None:
        path = go_source()

    for arg in args:
        path = Path.joinpath(Path(path), arg)

    return path


def get_path_data(directory: str = '', *args) -> Path:
    path_data = Path.joinpath(go_source(), 'data', directory)
    return get_path(path_data, *args)


def get_varname(self, variable: Any) -> str:
        local_variables = inspect.currentframe().f_back.f_locals.items()
        return [name for name, value in local_variables if value is variable][0]
