# utils.py

"""
My tools for data analysis
"""

from pathlib import Path
from typing import Annotated, Dict, Tuple, Callable, Any, List, Union
import re
import inspect
from itertools import groupby
from operator import itemgetter

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


def get_freq_perc(series: pd.Series) -> dict[str, tuple[int, float]]:
    series_dist = series.value_counts()
    series_perc = series.value_counts(normalize=True).values.round(4)
    return dict(
        zip(
            series_dist.keys(), 
            zip(series_dist.values, series_perc)
        )
    )


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


def extract_by_pos(data: Iterable, position: int = 0) -> Iterable:
    return [i for i, _ in groupby(data, itemgetter(position))]

    
def plot_bar(
    data: pd.Series, 
    figsize: Optional[tuple[float, float]] = None, 
    title: Optional[str] = None,
    save: bool = False,
    filename: Optional[str] = None,
    customize_color: bool = False, 
    **colors: Any
) -> None:
    
    fig, ax = plt.subplots(1, figsize=figsize)
    
    dist = get_freq_perc(data)
    _dist = extract_by_pos(dist.items(), 1)
    
    cat = extract_by_pos(dist.items(), 0)
    freq = extract_by_pos(_dist, 0)
    perc = extract_by_pos(_dist, 1)
    
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.xticks([])
    plt.yticks(fontsize=12, va='center')
    
    bars = plt.barh(cat, freq, alpha=0.8, height=0.45)
    
    if customize_color:
        assert(len(bars) == len(colors.values()))
        
        for bar, color in zip(bars, colors.values()):
            bar.set_color(color)
    
    for i, val in enumerate(zip(freq, perc)):
        plt.text(val[0] + 0.5, i, val[1], fontsize=12, va='center')
    
    plt.title(title)
    
    if save:
        if filename:
            plt.savefig(filename, transparent=True)
        else:
            plt.savefig("plot.png", transparent=True)
        
    plt.show()
