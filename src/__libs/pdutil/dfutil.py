from enum import Enum

import pandas as pd
import numpy as np


def renameCategoriesWithCounts(df: pd.DataFrame, uniqueCol: str, categoryCol: str, newLabelCol:str = None, pattern = "%s (%d)")->pd.DataFrame:
    """Rename the categories in the dataframe by appending their counts
    to the category name. 
    E.g. if the category is "A" and there are 10 instances of "A". A will be renamed to "A (10)"
    :param df: The dataframe to rename the categories in
    :param uniqueCol: The column that uniquely identifies the rows
    :param categoryCol: The column that contains the categories to rename
    :param newLabelCol: The column to store the new labels in. If None, the categoryCol will be overwritten.
    :param pattern: The pattern to use to rename the categories. It should contain two placeholders for the category name and the count.
    :return: The dataframe with the renamed categories
    """

    counts = df.groupby([uniqueCol, categoryCol]).count().reset_index()
    counts = counts[categoryCol].value_counts().to_dict()

    if newLabelCol is None:
        newLabelCol = categoryCol

    for k, v in counts.items():
        df.loc[df[categoryCol] == k, newLabelCol] = pattern%(k,v)

    return df


def filterDataFrame(data, **filters):
    if filters is None: return data

    data = data.copy()
    def parseFV(fv:any):

        if isinstance(fv, list):
            return [parseFV(v) for v in fv]

        if isinstance(fv, Enum): return fv.value
        if getattr(fv, "name", None) is not None: return fv.name

        return fv

    # Run all the filters
    for k, v in filters.items():
        parsedVals = parseFV(v)
        if isinstance(v, list):
            data = data.loc[data[k].isin(parsedVals)]
        else:
            data = data.loc[data[k] == parsedVals]

        assert len(data) > 0, f"Filter {k} with value {v} returned no results"

    filterString = " / ".join(str(parseFV(item)) for key,item in filters.items())

    return data,filterString