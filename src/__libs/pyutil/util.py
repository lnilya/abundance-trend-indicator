from typing import Literal

import pandas as pd
import os
import numpy as np


def excludeFromDF(df:pd.DataFrame,cond,msg,silent = False)->pd.DataFrame:
    """
    Exclude rows from a dataframe based on a condition
    :param df: Original dataframe
    :param cond: binary array. If true the values will be kept
    :param msg: Message to print informing about how many rows have been removed
    :return: the dataframe with the rows removed
    """
    ldf = len(df)
    df = df.loc[cond,:]
    if ldf != len(df) and not silent:
        print(f"    {msg}: Removed {ldf - len(df)} (~{100 * (ldf-len(df))/ldf:.2f} %) rows")
    return df


def runIfNotExists(overwrite, path, func, *args)->pd.DataFrame:
    """
    Run the given function if a file does not exist. Useful when the output of the function
    is written to a file.
    :param overwrite: If true will run regardless
    :param path: Path of file
    :param func: Function to run
    :param args: Parameters to pass
    :return:
    """
    if overwrite or not os.path.exists(path):
        return func(*args)
    else:
        print(f"Skipping {func.__name__} output already exists.")
        return pd.read_csv(path)


def subsampleAndAverage(arr, subsample:int=10):
    """Given a 2D array will subsample it given the integer factor and average. Cheap way to make images smaller."""
    # Get the original dimensions
    original_shape = arr.shape

    # Calculate the new dimensions
    new_width = original_shape[0] // subsample
    new_height = original_shape[1] // subsample

    # Reshape the array to group 10x10 pixels
    reshaped = arr[:new_width * subsample, :new_height * subsample].reshape(new_width, subsample, new_height, subsample)

    # Calculate the average of each group
    with np.errstate(invalid='ignore'):
        averaged = np.nanmean(reshaped,axis=(1,3))

    return averaged

def stackImages(imgs:list[np.ndarray], dir:Literal["h","v"])->np.ndarray:
    if dir == "h":
        return np.hstack(imgs)
    else:
        return np.vstack(imgs)

