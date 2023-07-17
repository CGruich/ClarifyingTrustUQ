"""
Visualizations for predictive uncertainties and metrics.
"""
from typing import Union, Tuple, List, Any, NoReturn
import pathlib

import numpy as np
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
)
from shapely.geometry import Polygon, LineString
from shapely.ops import polygonize, unary_union

from uncertainty_toolbox.metrics_calibration import (
    get_proportion_lists,
    get_proportion_lists_vectorized,
    adversarial_group_calibration,
)

from sklearn.isotonic import IsotonicRegression

# Save the unicode string for the lambda symbol so we can put it in the plots later.
# Make this a global variable such that it is accessible from all functions.
global lambdaSymbolStr
lambdaSymbolStr = "\u03bb"


def plot_accuracy_lineplot(
        error_array: np.ndarray,
        leg_loc: Union[int, str] = 2,
        leg_labels: Union[list, None] = None,
        colorList: list = ["#1f77b4", "#CD0000"],
        markerList: list = ['o', 's', 'P', '^'],
        lineList: list = ['dashed', 'dotted', 'dashdot', 'solid'],
        xLabels: Union[list, None] = None,
        showTitle: bool = True,
        showLegend: bool = True,
        ax: Union[matplotlib.axes.Axes, None] = None,
) -> matplotlib.axes.Axes:
    """ Plot a lineplot of ML model overall accuracy based on desired error metric.

    Args:
        error_array: 1D array of the error values of each model scenario (e.g. different learning rates)
                     Each entry in the 1D array is a tuple. A tuple of size 2 means plotting one line plot in (x, y) coordinates.
                     A tuple of size 3 means plotting two line pltos in (x, y, z) coordinates, where the first plot is specified by (x, y)
                     and the second plot is specified by (x, z)

        leg_loc: location of legend as a str or legend code int
        colorList: A list of string color values compatible with matplotlib. List is to be the same size of tuples in error_array.
        ax: matplotlib.axes.Axes object

    Returns:
        matplotlib.axes.Axes object with plot added."""

    # Create ax if it doesn't exist
    if ax is None:
        fix, ax = plt.subplots(figsize=(5, 5))

    xPoints = [points[0] for points in error_array]
    numPointsPerTrend = len(xPoints)
    trendData = [points[1:] for points in error_array]
    numLineTrends = len(trendData[0])
    # Construct the line plot
    for trendInd in range(0, numLineTrends):
        yPoints = [points[(trendInd + 1)] for points in error_array]
        numYPoints = len(yPoints)

        # Set up x-axis labeling
        if xLabels is not None:
            assert len(xLabels) == numPointsPerTrend
            plt.xticks(xPoints, xLabels)

        # Make sure we have enough colors for each trend
        assert len(colorList) >= numLineTrends

        # Make sure we have enough marker shapes for each trend
        assert len(markerList) >= numLineTrends

        if leg_labels is not None:
            ax.plot(xPoints, yPoints, color="black", marker=markerList[trendInd], markerfacecolor=colorList[trendInd], markeredgecolor='black',
                    markeredgewidth='2', linestyle=lineList[trendInd], linewidth=2, markersize=8, label=leg_labels[trendInd])
        elif leg_labels is None:
            ax.plot(xPoints, yPoints, color="black", marker=markerList[trendInd], markerfacecolor=colorList[trendInd], markeredgecolor='black',
                    markeredgewidth='2', linestyle=lineList[trendInd], linewidth=2, markersize=8)

    # Construct a legend if not None
    if (leg_labels is not None) and (showLegend == True):
        assert len(leg_labels) == numLineTrends
        ax.legend(labels=leg_labels)

    if xLabels != None:
        ax.set_xticks(xPoints, xLabels)

    ax.set_xlabel("Regularization Hyperparameter " + lambdaSymbolStr)
    ax.set_ylabel("Mean Absolute Error (eV)")
    if showTitle == True:
        ax.set_title(
            "Test Accuracy versus Regularization Hyperparameter\n(Evidential Regression)")
    return ax


def plot_accuracy_barchart(
        error_array: np.ndarray,
        leg_loc: Union[int, str] = 2,
        leg_labels: list = None,
        colorList: list = ["#1f77b4", "#CD0000"],
        xLabels: list = None,
        ax: Union[matplotlib.axes.Axes, None] = None,
) -> matplotlib.axes.Axes:
    """Plot a bar chart of ML model overall accuracy based on desired error metric.

    Args:
        error_array: 1D array of the error values of each model scenario (e.g. different learning rates)
                     Each entry in the 1D array is a tuple. A tuple of size 2 means plotting two bars side-by-side.
                     A tuple of size 3 means plotting three bars side-by-side, etc.

        leg_loc: location of legend as a str or legend code int
        colorList: A list of string color values compatible with matplotlib. List is to be the same size of tuples in error_array.
        ax: matplotlib.axes.Axes object

    Returns:
        matplotlib.axes.Axes object with plot added."""

    # Create ax if it doesn't exist
    if ax is None:
        fix, ax = plt.subplots(figsize=(10, 10))

    numBars = erroy_array.size
    xIndices = np.arange(numBars)

    xWidth = 0.35

    # Set up x-axis labeling
    if xLabels is not None:
        assert len(xLabels) == numBars

    # Make sure we have enough colors for each bar
    assert len(colorList) == len(error_array[0])

    # Construct the bar chart
    for xInd in range(xIndices.shape[0]):
        for barInd in range(len(xIndices[xInd])):
            ax.bar(xIndices[xInd], xIndicies[xInd][barInd],
                   color=colorList[barInd], width=xWidth)

    # Construct a legend if not None
    if (leg_labels is not None) and (showLegend == True):
        assert len(leg_labels) == len(colorList)
        ax.legend(labels=leg_labels)

    ax.set_xticks(xIndices, xLabels)
    ax.set_ylabel("Error")
    if showTitle == True:
        ax.set_title("Accuracy Bar Chart")
    return ax


def plot_intervals_ordered(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    lamb: float,
    n_subset: Union[int, None] = None,
    ylims: Union[Tuple[float, float], None] = None,
    num_stds_confidence_bound: int = 3,
    seed: Union[int, None] = None,
    showTitle: bool = True,
    showLegend: bool = True,
    aspect_ratio: Union[float, None] = None,
    horizontal: bool = True,
    markerSize: int = 10,
    ax: Union[matplotlib.axes.Axes, None] = None,
) -> matplotlib.axes.Axes:
    """Plot predictions and predictive intervals versus true values, with points ordered
    by true value along x-axis.

    Args:
        y_pred: 1D array of the predicted means for the held out dataset.
        y_std: 1D array of the predicted standard deviations for the held out dataset.
        y_true: 1D array of the true labels in the held out dataset.
        n_subset: Number of points to plot after filtering.
        ylims: a tuple of y axis plotting bounds, given as (lower, upper).
        num_stds_confidence_bound: width of intervals, in terms of number of standard
            deviations.
        ax: matplotlib.axes.Axes object.

    Returns:
        matplotlib.axes.Axes object with plot added.
    """
    # Create ax if it doesn't exist
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # Optionally select a subset
    if n_subset is not None:
        [y_pred, y_std, y_true] = filter_subset(
            [y_pred, y_std, y_true], n_subset, seed)

    order = np.argsort(y_pred.flatten())
    y_pred, y_std, y_true = y_pred[order], y_std[order], y_true[order]
    xs = np.arange(len(order))
    intervals = num_stds_confidence_bound * y_std

    # Plot
    if horizontal:
        h3 = ax.errorbar(
            xs,
            y_pred,
            intervals,
            fmt=".",
            ls="none",
            linewidth=2.5,
            c="#1f77b4",
            alpha=0.5,
        )
        h1 = ax.plot(xs, y_true, "*", mec="black", mfc="#CD0000",
                     markersize=markerSize, markeredgewidth=1.0)
        h2 = ax.plot(xs, y_pred, ".", mec="black", mfc="#1f77b4",
                     markersize=markerSize, markeredgewidth=1.0)

    elif not horizontal:
        h3 = ax.errorbar(
            y_pred,
            xs,
            xerr=intervals,
            fmt=".",
            ls="none",
            linewidth=2.5,
            c="#1f77b4",
            alpha=0.5,
        )
        h1 = ax.plot(y_true, xs, "*", mec="black", mfc="#CD0000",
                     markersize=markerSize, markeredgewidth=1.0)
        h2 = ax.plot(y_pred, xs, ".", mec="black", mfc="#1f77b4",
                     markersize=markerSize, markeredgewidth=1.0)

    # Legend
    if showLegend == True:
        if horizontal:
            ax.legend([h2[0], h1[0], h3[0]], ["Ground Truth", "Prediction", "+/- " +
                      str(num_stds_confidence_bound) + "\u03C3"], title=None, loc=2, frameon=False)
        if not horizontal:
            ax.legend([h2[0], h1[0], h3[0]], ["Ground Truth", "Prediction", "+/- " +
                      str(num_stds_confidence_bound) + "\u03C3"], title=None, loc=4, frameon=False)

    # Determine lims
    if ylims is None:
        intervals_lower = np.min(y_pred - intervals)
        intervals_upper = np.max(y_pred + intervals)
        y_true_lower = np.min(y_true)
        y_true_upper = np.max(y_true)

        minVals = [intervals_lower, y_true_lower]
        maxVals = [intervals_upper, y_true_upper]

        minVal = np.min(minVals)
        maxVal = np.max(maxVals)

        lims_ext = [
            (minVal - 0.02),
            (maxVal + 0.02)
        ]
    else:
        lims_ext = ylims

    # Format plot
    if horizontal:
        ax.set_ylim(lims_ext)
        ax.set_xlabel("Cluster Index\n(Ordered by Prediction)")
        ax.set_ylabel("Predicted\nAdsorption Energy (eV)")
    elif not horizontal:
        ax.set_xlim(lims_ext)
        ax.set_ylabel("Cluster Index\n(Ordered by Prediction)")
        ax.set_xlabel("Predicted Adsorption Energy (eV)")

    if showTitle == True:
        ax.set_title("Ordered Prediction Intervals (Evidential Regression, " +
                     lambdaSymbolStr + " = " + str(lamb) + ")")

    if aspect_ratio is not None:
        ax.set_aspect(aspect_ratio / ax.get_data_ratio(), adjustable="box")

    return ax


def filter_subset(input_list: List[List[Any]], n_subset: int, seed: Union[int, None] = None) -> List[List[Any]]:
    """Keep only n_subset random indices from all lists given in input_list.

    Args:
        input_list: list of lists.
        n_subset: Number of points to plot after filtering.

    Returns:
        List of all input lists with sizes reduced to n_subset.
    """
    assert type(n_subset) is int

    # Set random number generation seed for reproducibility, if applicable.
    if seed is not None:
        assert type(seed) is int
        np.random.seed(seed)

    n_total = len(input_list[0])
    idx = np.random.choice(range(n_total), n_subset, replace=False)
    idx = np.sort(idx)
    output_list = []
    for inp in input_list:
        outp = inp[idx]
        output_list.append(outp)
    return output_list


def set_style(style_str: str = "default") -> NoReturn:
    """Set the matplotlib plotting style.

    Args:
        style_str: string for style file.
    """
    if style_str == "default":
        plt.style.use(
            (pathlib.Path(__file__).parent / "matplotlibrc").resolve())


def save_figure(
    file_name: str = "figure",
    ext_list: Union[list, str, None] = None,
    white_background: bool = True,
) -> NoReturn:
    """Save matplotlib figure for all extensions in ext_list.

    Args:
        file_name: name of saved image file.
        ext_list: list of strings (or single string) denoting file type.
        white_background: set background of image to white if True.
    """

    # Default ext_list
    if ext_list is None:
        ext_list = ["pdf", "png"]

    # If ext_list is a single str
    if isinstance(ext_list, str):
        ext_list = [ext_list]

    # Set facecolor and edgecolor
    (fc, ec) = ("w", "w") if white_background else ("none", "none")

    # Save each type in ext_list
    for ext in ext_list:
        save_str = file_name + "." + ext
        plt.savefig(save_str, bbox_inches="tight", facecolor=fc, edgecolor=ec)
        print(f"Saved figure {save_str}")


def update_rc(key_str: str, value: Any) -> NoReturn:
    """Update matplotlibrc parameters.

    Args:
        key_str: string for a matplotlibrc parameter.
        value: associated value to set the matplotlibrc parameter.
    """
    plt.rcParams.update({key_str: value})


def determine_common_axes(axsGrid):
    """For showing multiple plots on a 1xN grid, find common x-axis and y-axis ranges
    for all the plots.

    Args:
        axsGrid: 1xN list of axes objects in a plot grid
    """

    # axsGrid: An axes subplot 1xN list for each subplot
    ylimsUpper = []
    ylimsLower = []
    xlimsUpper = []
    xlimsLower = []

    for ax in axsGrid:
        # Returns lists of either [bottom, top] or [left, right]
        ax_ylim = ax.get_ylim()
        ax_xlim = ax.get_xlim()

        ylimsUpper.append(ax_ylim[1])
        ylimsLower.append(ax_ylim[0])
        xlimsUpper.append(ax_xlim[1])
        xlimsLower.append(ax_xlim[0])

    common_ylimsUpper = max(ylimsUpper)
    common_ylimsLower = min(ylimsLower)
    common_xlimsUpper = max(xlimsUpper)
    common_xlimsLower = min(xlimsLower)

    common_ylims = (common_ylimsLower, common_ylimsUpper)
    common_xlims = (common_xlimsLower, common_xlimsUpper)

    common_lims = (common_xlims, common_ylims)

    return common_lims


def determine_common_colorbar(plot_data: Union[list, tuple]):
    """For showing multiple plots on a 1xN grid, determine a common color scheme for colorbars/heatmaps/etc.

    Args:
        plot_data: a list or tuple of N arrays of data used to generate the color bar or heat map
                    (e.g., a list of N arrays where each array is corresponds to individual bin counts in a hexbin plot)

    """

    # axsGrid: An axes subplot 1xN list for each subplot
    dataUpper = []
    dataLower = []

    for data in plot_data:
        dataMin = min(data)
        dataMax = max(data)

        dataLower.append(dataMin)
        dataUpper.append(dataMax)

    common_Upper = max(dataUpper)
    common_Lower = min(dataLower)

    common_lims = (common_Lower, common_Upper)

    return common_lims
