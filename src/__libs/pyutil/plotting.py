from typing import List

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np

def get2DMaxLikelihoodCovarianceEllipse(df:pd.DataFrame, columns:List[str], weights = None, numPoints = 100, numStd = 1):
    """
    Creates an ellipse from the weighted max likelihood approximation of 2D data, ready to be plotted
    :param df: Dataframe with float values
    :param columns: Exactly 2 columns to take from the datframe
    :param weights: Optional set of weights that will be used to weigh the max likelihood solution, if some points are more important than others
    :param numPoints: Number of points along the boundary of the ellipse
    :return: a 2xnumPoints array with the x and y coordinates of the ellipse
    """
    # Assuming grA is your data array and vars contains the indices of the variables you are interested in

    if weights is None:
        weights = np.ones(df.shape[0])

    weights /= np.sum(weights)

    points = np.array([df[columns[0]], df[columns[1]]]).T
    # Step 1: Calculate the weighted mean vector
    mean = np.average(points, axis=0, weights=weights)
    centered_points = points - mean
    mc = (centered_points.T @ np.diag(weights) @ centered_points)

    # Eigen decomposition of the covariance matrix
    eigvals, eigvecs = np.linalg.eigh(mc)

    # Eigenvalues and eigenvectors
    largest_eigval = eigvals[1]
    largest_eigvec = eigvecs[:, 1]
    smallest_eigval = eigvals[0]
    smallest_eigvec = eigvecs[:, 0]

    # Angle of rotation of ellipse
    theta = np.arctan2(largest_eigvec[1], largest_eigvec[0])

    # Create the ellipse
    phi = np.linspace(0, 2 * np.pi, numPoints)
    ellipse = np.array([np.cos(phi), np.sin(phi)])

    # Scaling and rotation
    scale_matrix = np.array([[np.sqrt(largest_eigval) * numStd, 0], [0, np.sqrt(smallest_eigval) * numStd]])
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    ellipse = rotation_matrix @ scale_matrix @ ellipse

    # Shift to the mean
    ellipse[0, :] += mean[0]
    ellipse[1, :] += mean[1]

    return ellipse

def setUpSubplotMatplotlib(rows: int = None, columns: int = None, windowTitle: str = None, axisTitles: List[str] = None, showAxis=True,
                           fullW: bool = False, fullH: bool = False, bgCol=None) -> List[Axes]:
    """
    Sets up matplotlibs with subplots, titles - convenience function to plot things
    :param rows: Number of rows
    :param columns: Number of columns
    :param windowTitle: Title of the matplotlib window
    :param axisTitles: Flat list of titles for each subplot
    :param showAxis: if true will show axis, if false will hide axis, if list will hide axis for each subplot
    :param fullW: Just the size, set arbitrary to 18.5"
    :param fullH: Just the size, set arbitrary to 10"
    :param bgCol: Background color for the plot
    :return: A flattened list (rows x cols) of axis object for adding the plots
    """
    fig, ax = plt.subplots(rows, columns)
    if bgCol is not None:
        fig.patch.set_facecolor(bgCol)
    if windowTitle is not None:
        fig.canvas.manager.set_window_title(windowTitle)

    # flatten axis object, makes it easier to handle
    if (rows + columns > 2):
        ax = ax.ravel()
    else:
        ax = [ax]

    # display titles if desired
    if axisTitles is not None:
        for i, a in enumerate(ax):
            if (i >= len(axisTitles)): break
            a.set_title(axisTitles[i])

    # display axises if desired
    if showAxis is False:
        [a.axis('off') for a in ax]


    elif isinstance(showAxis, list):
        for i, a in enumerate(ax):
            if showAxis[i]: continue
            a.axis('off')

    # size of widow
    fig.set_size_inches(18.5 if fullW else 6, 10 if fullH else 4.5)
    plt.tight_layout()
    return ax