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
lambdaSymbolStr = '\u03bb'


def plot_accuracy_lineplot(
    error_array: np.ndarray,
    leg_loc: Union[int, str] = 2,
    leg_labels: Union[list, None] = None,
    colorList: list = ['#1f77b4', '#CD0000'],
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
            ax.plot(
                xPoints,
                yPoints,
                color='black',
                marker=markerList[trendInd],
                markerfacecolor=colorList[trendInd],
                markeredgecolor='black',
                markeredgewidth='2',
                linestyle=lineList[trendInd],
                linewidth=2,
                markersize=8,
                label=leg_labels[trendInd],
            )
        elif leg_labels is None:
            ax.plot(
                xPoints,
                yPoints,
                color='black',
                marker=markerList[trendInd],
                markerfacecolor=colorList[trendInd],
                markeredgecolor='black',
                markeredgewidth='2',
                linestyle=lineList[trendInd],
                linewidth=2,
                markersize=8,
            )

    # Construct a legend if not None
    if (leg_labels is not None) and (showLegend == True):
        assert len(leg_labels) == numLineTrends
        ax.legend(labels=leg_labels)

    if xLabels != None:
        ax.set_xticks(xPoints, xLabels)

    ax.set_xlabel('Training Dropout Rate')
    ax.set_ylabel('Mean Absolute Error (eV)')
    if showTitle == True:
        ax.set_title('Test Accuracy versus Dropout Rate (MC Dropout)')
    return ax


def plot_accuracy_barchart(
    error_array: np.ndarray,
    leg_loc: Union[int, str] = 2,
    leg_labels: list = None,
    colorList: list = ['#1f77b4', '#CD0000'],
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
            ax.bar(
                xIndices[xInd], xIndicies[xInd][barInd], color=colorList[barInd], width=xWidth
            )

    # Construct a legend if not None
    if (leg_labels is not None) and (showLegend == True):
        assert len(leg_labels) == len(colorList)
        ax.legend(labels=leg_labels)

    ax.set_xticks(xIndices, xLabels)
    ax.set_ylabel('Error')
    if showTitle == True:
        ax.set_title('Accuracy Bar Chart')
    return ax


def plot_parity(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    n_subset: Union[int, None] = None,
    ylims: Union[Tuple[float, float], None] = None,
    xlims: Union[Tuple[float, float], None] = None,
    leg_loc: Union[int, str] = 4,
    seed: Union[int, None] = None,
    showTitle: bool = True,
    showLegend: bool = True,
    ax: Union[matplotlib.axes.Axes, None] = None,
) -> matplotlib.axes.Axes:
    """ Plot one-dimensional predictions with one-dimensional observations (i.e. true values).

    Args:
        y_pred: 1D array of the predicted means for the held out dataset.
        y_std: 1D array of the predicted standard deviations for the held out dataset.
        y_true: 1D array of the true labels in the held out dataset.
        n_subset: Number of points to plot after filtering
        ylims: a tuple of y axis plotting bounds, given as (lower, upper).
        xlims: a tuple of x axis plotting bounds, given as (lower, upper).
        leg_loc: location of legend as a str or legend code int.
        ax: matplotlib.axes.Axes object

        Returns:
            matplotlib.axes.Axes object with plot added.
    """

    # Create ax if it doesn't exist
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # Order points in order of increasing y_true
    order = np.argsort(y_true)
    y_pred, y_std, y_true = (
        y_pred[order],
        y_std[order],
        y_true[order],
    )

    # Optionally select a subset
    if n_subset is not None:
        [y_pred, y_std, y_true] = filter_subset(
            [y_pred, y_std, y_true], n_subset, seed)

    # Identity line
    h2 = ax.plot([0, 1], [0, 1], '--', transform=ax.transAxes,
                 c='#CD0000', linewidth=2)
    # Parity plot points
    h1 = ax.plot(y_true, y_pred, '.', mec='black',
                 mfc='#1f77b4', markeredgewidth=1.0)

    if showLegend == True:
        ax.legend(
            [h2[0], h1[0]],
            ['Perfect Fit', 'Prediction'],
            title=str(n_subset) + ' Clusters',
            loc=leg_loc,
        )

    # Determine lims

    if ylims is None:
        lower_upper_ylims = [y_pred, y_true]
        lims_ext_ylims = [
            int(np.floor(np.min(lower_upper_ylims))),
            int(np.ceil(np.max(lower_upper_ylims))),
        ]
    else:
        lims_ext_ylims = ylims

    if xlims is None:
        lower_upper_xlims = [y_pred, y_true]
        lims_ext_xlims = [
            int(np.floor(np.min(lower_upper_xlims))),
            int(np.ceil(np.max(lower_upper_xlims))),
        ]
    else:
        lims_ext_xlims = xlims

    # Format plot
    ax.set_xlim(lims_ext_xlims)
    ax.set_ylim(lims_ext_ylims)
    ax.set_xlabel('Adsorption Energy (eV)')
    ax.set_ylabel('Predicted\nAdsorption Energy (eV)')

    if showTitle == True:
        ax.set_title('Parity Plot (MC Dropout)')
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')

    return ax


def plot_parity_hexagonal(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    gridSize: int = 100,
    bins: Union[str, None] = None,
    subFig: Union[matplotlib.pyplot.figure, None] = None,
    n_subset: Union[int, None] = None,
    ylims: Union[Tuple[float, float], None] = None,
    xlims: Union[Tuple[float, float], None] = None,
    leg_loc: Union[int, str] = 4,
    seed: Union[int, None] = None,
    colorRange: Union[list, tuple, None] = None,
    showTitle: bool = True,
    showLegend: bool = True,
    showColorBar: bool = True,
    getCounts: bool = False,
    getLims: bool = False,
    statLabels: Union[list, None] = None,
    ax: Union[matplotlib.axes.Axes, None] = None,
) -> matplotlib.axes.Axes:
    """ Plot one-dimensional predictions with one-dimensional observations (i.e. true values).

    Args:
        y_pred: 1D array of the predicted means for the held out dataset.
        y_std: 1D array of the predicted standard deviations for the held out dataset.
        y_true: 1D array of the true labels in the held out dataset.
        gridSize: Amount of hexagons on the x-axis to plot. Usually, this is auto-matched on the y-axis or some compatible arrangement is automatically determined.
        bins: Whether to calculate hexgonal bin counts based on count, log scale, etc.
        subFig: To plot a color bar with the hexagonal bin, we need the figure object of our plot. Pass this here.
        n_subset: Number of points to plot after filtering
        ylims: a tuple of y axis plotting bounds, given as (lower, upper).
        xlims: a tuple of x axis plotting bounds, given as (lower, upper).
        leg_loc: location of legend as a str or legend code int.
        ax: matplotlib.axes.Axes object

        Returns:
            matplotlib.axes.Axes object with plot added.
    """

    # Create ax if it doesn't exist
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # Order points in order of increasing y_true
    order = np.argsort(y_true)
    y_pred, y_std, y_true = (
        y_pred[order],
        y_std[order],
        y_true[order],
    )

    # Color range for heatmaps and color bars is specified as a list or tuple of size 2, (vmin, vmax)
    # Ensure that the list or tuple is of size 2 is an explicit color mapping is specified.
    if colorRange != None:
        assert len(colorRange) == 2

    # Optionally select a subset
    if n_subset is not None:
        [y_pred, y_std, y_true] = filter_subset(
            [y_pred, y_std, y_true], n_subset, seed)

    # Determine lims

    if ylims is None:
        lower_upper_ylims = [y_pred, y_true]
        lims_ext_ylims = [
            int(np.floor(np.min(lower_upper_ylims))),
            int(np.ceil(np.max(lower_upper_ylims))),
        ]
    else:
        lims_ext_ylims = ylims

    if xlims is None:
        lower_upper_xlims = [y_pred, y_true]
        lims_ext_xlims = [
            int(np.floor(np.min(lower_upper_xlims))),
            int(np.ceil(np.max(lower_upper_xlims))),
        ]
    else:
        lims_ext_xlims = xlims

    # Identity line
    h2 = ax.plot([0, 1], [0, 1], '--', transform=ax.transAxes,
                 c='#CD0000', linewidth=2)
    # Parity plot points, expressed as hexagonal binning
    # h1 = ax.plot(y_true, y_pred, ".", mec="#1f77b4", mfc="#1f77b4")

    if colorRange != None:
        h1 = ax.hexbin(
            y_true,
            y_pred,
            gridsize=gridSize,
            bins=bins,
            extent=(
                lims_ext_xlims[0],
                lims_ext_xlims[1],
                lims_ext_ylims[0],
                lims_ext_ylims[1],
            ),
            cmap='Blues',
            vmin=colorRange[0],
            vmax=colorRange[1],
            linewidths=0.1,
        )
    else:
        h1 = ax.hexbin(
            y_true,
            y_pred,
            gridsize=gridSize,
            bins=bins,
            extent=(
                lims_ext_xlims[0],
                lims_ext_xlims[1],
                lims_ext_ylims[0],
                lims_ext_ylims[1],
            ),
            cmap='Blues',
            linewidths=0.1,
        )

    counts = h1.get_array()

    if showColorBar == True:
        cb = subFig.colorbar(h1, ax=ax, pad=0.04)
        if bins == None:
            cb.set_label('Samples')
        elif bins == 'log':
            cb.set_label('log10(Samples)')

    if n_subset is not None:
        titleName = str(n_subset) + ' Clusters'
    else:
        titleName = None
    if showLegend == True:
        ax.legend(
            [h2[0]], ['Perfect Fit'], title=titleName, loc=leg_loc,
        )

    # Format plot
    ax.set_xlim(lims_ext_xlims)
    ax.set_ylim(lims_ext_ylims)
    ax.set_xlabel('Adsorption Energy (eV)')
    ax.set_ylabel('Predicted\nAdsorption Energy (eV)')
    if showTitle == True:
        ax.set_title('Parity Plot (MC Dropout)')
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')

    # Show any quantitative accuracy metrics
    if statLabels is not None:
        # Show accuracy metrics (e.g., MAE) in a textbox.
        accuracyStr = ""
        for ind in range(len(statLabels)):
            subStr = statLabels[ind][0] + ' = ' + statLabels[ind][1] + '\n'
            accuracyStr = accuracyStr + subStr
        ax.text(
            x=0.02,
            y=0.98,
            s=accuracyStr,
            verticalalignment='top',
            horizontalalignment='left',
            fontsize=8,
            transform=ax.transAxes,
        )

    # If we want to get the plot limits,
    if getLims == True:
        if getCounts == True:
            # Store the plot limits has a tuple of lists
            limTuple = (list(ax.get_xlim()), list(ax.get_ylim()))
            axTuple = (ax, counts, limTuple)
            return axTuple
        else:
            axTuple = (ax, limTuple)
            return axTuple
    # If we only want to get the counts in each hexagonal bin,
    elif getCounts == True:
        axTuple = (ax, counts)
        return axTuple
    else:
        return ax


def plot_xy(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    x: np.ndarray,
    n_subset: Union[int, None] = None,
    ylims: Union[Tuple[float, float], None] = None,
    xlims: Union[Tuple[float, float], None] = None,
    num_stds_confidence_bound: int = 2,
    leg_loc: Union[int, str] = 3,
    ax: Union[matplotlib.axes.Axes, None] = None,
) -> matplotlib.axes.Axes:
    """Plot one-dimensional inputs with associated predicted values, predictive
    uncertainties, and true values.

    Args:
        y_pred: 1D array of the predicted means for the held out dataset.
        y_std: 1D array of the predicted standard deviations for the held out dataset.
        y_true: 1D array of the true labels in the held out dataset.
        x: 1D array of input values for the held out dataset.
        n_subset: Number of points to plot after filtering.
        ylims: a tuple of y axis plotting bounds, given as (lower, upper).
        xlims: a tuple of x axis plotting bounds, given as (lower, upper).
        num_stds_confidence_bound: width of confidence band, in terms of number of
            standard deviations.
        leg_loc: location of legend as a str or legend code int.
        ax: matplotlib.axes.Axes object.

    Returns:
        matplotlib.axes.Axes object with plot added.
    """
    # Create ax if it doesn't exist
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # Order points in order of increasing x
    order = np.argsort(x)
    y_pred, y_std, y_true, x = (
        y_pred[order],
        y_std[order],
        y_true[order],
        x[order],
    )

    # Optionally select a subset
    if n_subset is not None:
        [y_pred, y_std, y_true, x] = filter_subset(
            [y_pred, y_std, y_true, x], n_subset)

    intervals = num_stds_confidence_bound * y_std

    h1 = ax.plot(x, y_true, '.', mec='#1f77b4', mfc='None')
    h2 = ax.plot(x, y_pred, '--', c='#CD0000', linewidth=2)
    h3 = ax.fill_between(
        x, y_pred - intervals, y_pred + intervals, color='lightsteelblue', alpha=0.4,
    )
    ax.legend(
        [h1[0], h2[0], h3], ['Observations', 'Predictions', '$95\%$ Interval'], loc=leg_loc,
    )

    # Format plot
    if ylims is not None:
        ax.set_ylim(ylims)

    if xlims is not None:
        ax.set_xlim(xlims)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('Confidence Band')
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')

    return ax


def plot_intervals(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    n_subset: Union[int, None] = None,
    ylims: Union[Tuple[float, float], None] = None,
    num_stds_confidence_bound: int = 2,
    aspect_ratio: Union[float, None] = None,
    ax: Union[matplotlib.axes.Axes, None] = None,
) -> matplotlib.axes.Axes:
    """Plot predictions and predictive intervals versus true values.

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
            [y_pred, y_std, y_true], n_subset)

    # Compute intervals
    intervals = num_stds_confidence_bound * y_std

    # Plot
    ax.errorbar(
        y_true, y_pred, intervals, fmt='o', ls='none', linewidth=1.5, c='#1f77b4', alpha=0.5,
    )
    h1 = ax.plot(y_true, y_pred, 'o', c='#1f77b4')

    # Determine lims
    if ylims is None:
        intervals_lower_upper = [y_pred - intervals, y_pred + intervals]
        lims_ext = [
            int(np.floor(np.min(intervals_lower_upper[0]))),
            int(np.ceil(np.max(intervals_lower_upper[1]))),
        ]
    else:
        lims_ext = ylims

    # plot 45-degree line
    h2 = ax.plot(lims_ext, lims_ext, '--', linewidth=1.5, c='#CD0000')

    # Legend
    ax.legend([h1[0], h2[0]], ['Predictions', '$f(x) = x$'], loc=4)

    # Format plot
    ax.set_xlim(lims_ext)
    ax.set_ylim(lims_ext)
    ax.set_xlabel('Observed Values')
    ax.set_ylabel('Predicted Values and Intervals')
    ax.set_title('Prediction Intervals')

    if aspect_ratio is not None:
        ax.set_aspect(aspect_ratio / ax.get_data_ratio(), adjustable='box')

    return ax


def plot_intervals_ordered(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    n_subset: Union[int, None] = None,
    ylims: Union[Tuple[float, float], None] = None,
    num_stds_confidence_bound: int = 2,
    seed: Union[int, None] = None,
    showTitle: bool = True,
    showLegend: bool = True,
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

    order = np.argsort(y_true.flatten())
    y_pred, y_std, y_true = y_pred[order], y_std[order], y_true[order]
    xs = np.arange(len(order))
    intervals = num_stds_confidence_bound * y_std

    # Plot
    ax.errorbar(
        xs, y_pred, intervals, fmt='.', ls='none', linewidth=1.5, c='#1f77b4', alpha=0.5,
    )
    h1 = ax.plot(xs, y_pred, '.', mec='black',
                 mfc='#1f77b4', markeredgewidth=1.0)
    h2 = ax.plot(xs, y_true, '--', linewidth=2.0, c='#CD0000')

    # Legend
    if showLegend == True:
        ax.legend(
            [h2[0], h1[0]],
            ['Perfect Fit', 'Prediction'],
            title=str(n_subset) + ' Clusters',
            loc=4,
        )

    # Determine lims
    if ylims is None:
        intervals_lower_upper = [y_pred - intervals, y_pred + intervals]
        lims_ext = [
            int(np.floor(np.min(intervals_lower_upper[0]))),
            int(np.ceil(np.max(intervals_lower_upper[1]))),
        ]
    else:
        lims_ext = ylims

    # Format plot
    ax.set_ylim(lims_ext)
    ax.set_xlabel('Cluster Index\n(Ordered by Ground Truth Cluster)')
    ax.set_ylabel('Predicted\nAdsorption Energy\n(eV, +/- 2\u03C3)')
    if showTitle == True:
        ax.set_title('Ordered Prediction Intervals (MC Dropout)')
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')

    return ax


def plot_calibration(
    y_pred: Union[np.ndarray, list],
    y_std: Union[np.ndarray, list],
    y_true: np.ndarray,
    leg_loc: Union[int, str] = 4,
    leg_labels: Union[list, None] = None,
    colorList: list = ['#1f77b4', 'darkorange',
                       'darkgreen', 'gold', 'chocolate', 'darkcyan'],
    markerList: list = ['o', 's', 'P', '^', 'D', 'p'],
    lineList: list = [
        'dashed',
        'dotted',
        'dashdot',
        'solid',
        (0, (3, 1, 1, 1, 1, 1)),
        (0, (3, 5, 1, 5, 1, 5)),
    ],
    legendTitle: str = 'Miscalibration Area',
    n_subset: Union[int, None] = None,
    curve_label: Union[str, None] = None,
    show: bool = False,
    vectorized: bool = True,
    exp_props: Union[np.ndarray, list, None] = None,
    obs_props: Union[np.ndarray, list, None] = None,
    seed: Union[int, None] = None,
    showTitle: bool = True,
    showLegend: bool = True,
    showAlpha: bool = True,
    alphaColorMap: bool = False,
    legendAlign: str = 'left',
    recalModel: Union[list, IsotonicRegression] = None,
    ax: Union[matplotlib.axes.Axes, None] = None,
) -> matplotlib.axes.Axes:
    """Plot the observed proportion vs prediction proportion of outputs falling into a
    range of intervals, and display miscalibration area.

    Args:
        y_pred: 1D array of the predicted means for the held out dataset.
        y_std: 1D array of the predicted standard deviations for the held out dataset.
        y_true: 1D array of the true labels in the held out dataset.
        n_subset: Number of points to plot after filtering.
        curve_label: legend label str for calibration acs catalysis impact factorcurve.
        vectorized: plot using get_proportion_lists_vectorized.
        exp_props: plot using the given expected proportions.
        obs_props: plot using the given observed proportions.
        ax: matplotlib.axes.Axes object.

    Returns:
        matplotlib.axes.Axes object with plot added.
    """
    # Create ax if it doesn't exist
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # Store the line plot objects for later formatting
    plotObjects = []
    # Store the filled area objects of the plots for later formatting
    fillObjects = []
    # Save the data proportions used to draw the line plot objects for later formatting
    exp_props_list = []
    obs_props_list = []

    # If the calibration curve has multiple curves,
    if type(y_pred) is list and type(y_std) is list:
        # Optionally select a subset
        if n_subset is not None:
            filteredDatasets = []
            for trendInd in range(len(y_std)):
                filteredDataset = filter_subset(
                    [y_pred[trendInd], y_std[trendInd],
                        y_true[trendInd]], n_subset, seed
                )
                filteredDatasets.append(filteredDataset)
        # If proportion data is not specified manually to draw the calibration curve(s), then calculate the proportion data
        if (exp_props is None) or (obs_props is None):
            for trendInd in range(len(y_std)):
                # Compute exp_proportions and obs_proportions
                if vectorized:
                    (
                        exp_proportions_iter,
                        obs_proportions_iter,
                    ) = get_proportion_lists_vectorized(
                        y_pred[trendInd],
                        y_std[trendInd],
                        y_true,
                        recal_model=recalModel[trendInd],
                    )

                    exp_props_list.append(exp_proportions_iter)
                    obs_props_list.append(obs_proportions_iter)
                else:
                    (exp_proportions_iter, obs_proportions_iter) = get_proportion_lists(
                        y_pred[trendInd],
                        y_std[trendInd],
                        y_true,
                        recal_model=recalModel[trendInd],
                    )

                    exp_props_list.append(exp_proportions_iter)
                    obs_props_list.append(obs_proportions_iter)
        # If proportion data is already specified to draw the calibration curve(s), then format the proportion data for plotting
        else:
            for trendInd in range(len(y_std)):
                # If expected and observed proportions are given
                exp_proportions_iter = np.array(exp_props).flatten()
                obs_proportions_iter = np.array(obs_props).flatten()
                if exp_proportions_iter.shape != obs_proportions_iter.shape:
                    raise RuntimeError(
                        'exp_props and obs_props shape mismatch')

                exp_props_list.append(exp_proportions_iter)
                obs_props_list.append(obs_proportions_iter)

    # If the calibration curve only has one curve
    else:
        # Optionally select a subset
        if n_subset is not None:
            [y_pred, y_std, y_true] = filter_subset(
                [y_pred, y_std, y_true], n_subset, seed)

        # If proportion data is not specified manually to draw the calibration curve(s), then calculate the proportion data
        if (exp_props is None) or (obs_props is None):
            # Compute exp_proportions and obs_proportions
            if vectorized:
                (exp_proportions, obs_proportions,) = get_proportion_lists_vectorized(
                    y_pred, y_std, y_true, recal_model=recalModel
                )

                exp_props_list.append(exp_proportions)
                obs_props_list.append(obs_proportions)
            else:
                (exp_proportions, obs_proportions) = get_proportion_lists(
                    y_pred, y_std, y_true, recal_model=recalModel
                )

                exp_props_list.append(exp_proportions)
                obs_props_list.append(obs_proportions)
        # If proportion data is already specified to draw the calibration curve(s), then format the proportion data for plotting
        else:
            # If expected and observed proportions are given
            exp_proportions = np.array(exp_props).flatten()
            obs_proportions = np.array(obs_props).flatten()
            if exp_proportions.shape != obs_proportions.shape:
                raise RuntimeError('exp_props and obs_props shape mismatch')

            exp_props_list.append(exp_proportions)
            obs_props_list.append(obs_proportions)

    # Set label
    if curve_label is None:
        curve_label = 'Predictor'
    # Plot the perfect calibration line
    ax.plot([0, 1], [0, 1], '--', label='Ideal', c='#CD0000')

    # If we want to plot multiple calibration curves,
    if type(y_std) is list:
        for trendInd in range(len(y_std)):
            if alphaColorMap == False:
                (h1,) = ax.plot(
                    exp_props_list[trendInd],
                    obs_props_list[trendInd],
                    label=curve_label,
                    c=colorList[trendInd],
                )
            else:
                (h1,) = ax.plot(
                    exp_props_list[trendInd], obs_props_list[trendInd], label=curve_label
                )
            plotObjects.append(h1)
    # If we only want to plot one calibration curve,
    elif type(y_std) is not list:
        # Plot
        (h1,) = ax.plot(exp_props_list[0],
                        obs_props_list[0], label=curve_label)
        plotObjects.append(h1)

    # Format plot
    ax.set_xlabel('Expected Cumulative Distribution')
    ax.set_ylabel('Observed Cumulative Distribution')
    ax.axis('square')

    buff = 0.01
    ax.set_xlim([0 - buff, 1 + buff])
    ax.set_ylim([0 - buff, 1 + buff])

    if showTitle == True and type(y_std) is not list:
        ax.set_title('Average Calibration\n(5-fold Ensemble)')
    elif showTitle == True and type(y_std) is list:
        ax.set_title('Average Calibration')

    # Compute miscalibration area list
    miscalibration_areas = []

    # If we want to plot multiple calibration curves, then get the miscalibration area for each plot and add to list
    if type(y_std) is list:
        for trendInd in range(len(y_std)):
            polygon_points = []
            for point in zip(exp_props_list[trendInd], obs_props_list[trendInd]):
                polygon_points.append(point)
            for point in zip(
                reversed(exp_props_list[trendInd]), reversed(
                    exp_props_list[trendInd])
            ):
                polygon_points.append(point)
            polygon_points.append(
                (exp_props_list[trendInd][0], obs_props_list[trendInd][0]))
            polygon = Polygon(polygon_points)
            x, y = polygon.exterior.xy  # original data
            ls = LineString(np.c_[x, y])  # closed, non-simple
            lr = LineString(ls.coords[:] + ls.coords[0:1])
            mls = unary_union(lr)
            polygon_area_list = [poly.area for poly in polygonize(mls)]
            miscalibration_area = np.asarray(polygon_area_list).sum()

            miscalibration_areas.append(miscalibration_area)
    # Otherwise, if we want to plot one calibration curve, then get the miscalibration area for that one curve.
    elif type(y_std) is not list:
        polygon_points = []
        for point in zip(exp_props_list[0], obs_props_list[0]):
            polygon_points.append(point)
        for point in zip(reversed(exp_props_list[0]), reversed(exp_props_list[0])):
            polygon_points.append(point)
        polygon_points.append((exp_props_list[0][0], obs_props_list[0][0]))
        polygon = Polygon(polygon_points)
        x, y = polygon.exterior.xy  # original data
        ls = LineString(np.c_[x, y])  # closed, non-simple
        lr = LineString(ls.coords[:] + ls.coords[0:1])
        mls = unary_union(lr)
        polygon_area_list = [poly.area for poly in polygonize(mls)]
        miscalibration_area = np.asarray(polygon_area_list).sum()

        miscalibration_areas.append(miscalibration_area)

    # Put the miscalibration area(s) of the calibration curve(s) in the legend labels.
    # Store the labels in a list
    leg_labels_with_miscalibration = []
    # If we want legend labels at all (which we might not if we are plotting only one calibration curve),
    # then sort the legend labels, line plots, data proportions for plotting, and the colors based on
    # ascending miscalibraton area.
    if leg_labels != None and alphaColorMap == False:
        legendLabelsSortInfo = sorted(
            zip(
                miscalibration_areas,
                plotObjects,
                leg_labels,
                exp_props_list,
                obs_props_list,
                colorList,
            )
        )
        miscalibration_areas = [
            area for area, plot, label, exp_props, obs_props, color in legendLabelsSortInfo
        ]
        plotObjects = [
            plot for area, plot, label, exp_props, obs_props, color in legendLabelsSortInfo
        ]
        leg_labels = [
            label for area, plot, label, exp_props, obs_props, color in legendLabelsSortInfo
        ]
        exp_props_list = [
            exp_props
            for area, plot, label, exp_props, obs_props, color in legendLabelsSortInfo
        ]
        obs_props_list = [
            obs_props
            for area, plot, label, exp_props, obs_props, color in legendLabelsSortInfo
        ]
        colorList = [
            color for area, plot, label, exp_props, obs_props, color in legendLabelsSortInfo
        ]
    # Otherwise if we do not want a legend label (which we might not if we are plotting only one calibration curve),
    # then sort everything except the legend labels
    elif leg_labels == None and alphaColorMap == False:
        legendLabelsSortInfo = sorted(
            zip(miscalibration_areas, plotObjects,
                exp_props_list, obs_props_list, colorList)
        )
        miscalibration_areas = [
            area for area, plot, exp_props, obs_props, color in legendLabelsSortInfo
        ]
        plotObjects = [
            plot for area, plot, exp_props, obs_props, color in legendLabelsSortInfo
        ]
        exp_props_list = [
            exp_props for area, plot, exp_props, obs_props, color in legendLabelsSortInfo
        ]
        obs_props_list = [
            obs_props for area, plot, exp_props, obs_props, color in legendLabelsSortInfo
        ]
        colorList = [color for area, plot, exp_props,
                     obs_props, color in legendLabelsSortInfo]
    if leg_labels != None and alphaColorMap == True:
        legendLabelsSortInfo = sorted(
            zip(miscalibration_areas, plotObjects,
                leg_labels, exp_props_list, obs_props_list)
        )
        miscalibration_areas = [
            area for area, plot, label, exp_props, obs_props in legendLabelsSortInfo
        ]
        plotObjects = [
            plot for area, plot, label, exp_props, obs_props in legendLabelsSortInfo
        ]
        leg_labels = [
            label for area, plot, label, exp_props, obs_props in legendLabelsSortInfo
        ]
        exp_props_list = [
            exp_props for area, plot, label, exp_props, obs_props in legendLabelsSortInfo
        ]
        obs_props_list = [
            obs_props for area, plot, label, exp_props, obs_props in legendLabelsSortInfo
        ]
    # Otherwise if we do not want a legend label (which we might not if we are plotting only ne calibration curve),
    # then sort everything except the legend labels
    elif leg_labels == None and alphaColorMap == True:
        legendLabelsSortInfo = sorted(
            zip(miscalibration_areas, plotObjects,
                exp_props_list, obs_props_list)
        )
        miscalibration_areas = [
            area for area, plot, exp_props, obs_props in legendLabelsSortInfo
        ]
        plotObjects = [plot for area, plot, exp_props,
                       obs_props in legendLabelsSortInfo]
        exp_props_list = [
            exp_props for area, plot, exp_props, obs_props in legendLabelsSortInfo
        ]
        obs_props_list = [
            obs_props for area, plot, exp_props, obs_props in legendLabelsSortInfo
        ]

    # If we are plotting multiple calibration curves,
    if type(y_std) is list:
        for trendInd in range(len(y_std)):
            # If we are showing a filled in area for each calibration curve that are semi-transparent,
            # then we do not want the different curve colors to blend together and be aesthetically displeasing.

            # We already sorted the calibration curves by ascending miscalibration area,
            # So loop through and only fill in the miscalibration area between calibration curves. This prevents undesirable color blending.
            if showAlpha == True and trendInd == 0:
                if alphaColorMap == True:
                    h2 = ax.fill_between(
                        exp_props_list[trendInd],
                        exp_props_list[trendInd],
                        obs_props_list[trendInd],
                        alpha=0.2,
                        color=colorList[trendInd],
                    )
                else:
                    h2 = ax.fill_between(
                        exp_props_list[trendInd],
                        exp_props_list[trendInd],
                        obs_props_list[trendInd],
                        alpha=0.2,
                        color=colorList[trendInd],
                    )
                fillObjects.append(h2)
            # If we are plotting any curve that is not the first calibration curve,
            # then fill in the area between curves.
            # obs_props_list[trendInd - 1], obs_props_list[trendInd] inside ax.fill_between() does this
            elif showAlpha == True and trendInd > 0:
                if alphaColorMap == False:
                    h2 = ax.fill_between(
                        exp_props_list[trendInd],
                        obs_props_list[trendInd - 1],
                        obs_props_list[trendInd],
                        alpha=0.2,
                        color=colorList[trendInd],
                    )
                else:
                    h2 = ax.fill_between(
                        exp_props_list[trendInd],
                        obs_props_list[trendInd - 1],
                        obs_props_list[trendInd],
                        alpha=0.2,
                    )
                fillObjects.append(h2)
            else:
                h2 = ax.fill_between(
                    exp_props_list[trendInd],
                    exp_props_list[trendInd],
                    obs_props_list[trendInd],
                    alpha=0.0,
                    color=colorList[trendInd],
                )
                fillObjects.append(h2)
    # If we are only plotting one calibration curve,
    else:
        # If we are showing the filled in area of the calibration curve,
        if showAlpha == True:
            h2 = ax.fill_between(
                exp_props_list[0], exp_props_list[0], obs_props_list[0], alpha=0.2
            )
            fillObjects.append(h2)
        else:
            h2 = ax.fill_between(
                exp_props_list[0], exp_props_list[0], obs_props_list[0], alpha=0.0
            )
            fillObjects.append(h2)

    if alphaColorMap == True:
        # miscalibrationAreaColorMap = plt.cm.ScalarMappable(cmap="Blues", norm=matplotlib.colors.Normalize(vmin=miscalibration_areas[-1], vmax=miscalibration_areas[0]))
        miscalibrationAreaColorMap = matplotlib.cm.get_cmap('Blues_r')
        colorInterceptFactor = 0.15
        colorNorm = matplotlib.colors.Normalize(
            vmin=miscalibration_areas[0], vmax=0.5 + colorInterceptFactor
        )

        miscalibrationAreaNormList = []
        miscalibrationColorList = []
        miscalibrationAlphaList = []
        alphaScaleFactor = 0.8

        for trendInd in range(len(plotObjects)):
            miscalibrationAreaNormList.append(
                colorNorm(miscalibration_areas[trendInd]))
            miscalibrationAlpha = alphaScaleFactor * \
                (1 - miscalibrationAreaNormList[trendInd])
            miscalibrationAlphaList.append(miscalibrationAlpha)

            miscalibrationColorList.append(
                miscalibrationAreaColorMap(
                    miscalibrationAreaNormList[trendInd])
            )
            if trendInd == 0:
                plotObjects[trendInd].set_color('black')
            if trendInd > 0:
                plotObjects[trendInd].set_linestyle('None')
                plotObjects[trendInd].set(alpha=0.0)

        for trendInd in range(len(fillObjects)):
            fillObjects[trendInd].set_color(miscalibrationColorList[trendInd])
            fillObjects[trendInd].set(alpha=miscalibrationAlphaList[trendInd])

    # If a legend is being plotted for multiple calibration curves,
    if showLegend == True and type(y_std) is list:
        for trendInd in range(len(y_std)):
            # New legend label that includes the miscalibration area
            newLabel = '(' + leg_labels[trendInd] + \
                ', %.2f)' % miscalibration_areas[trendInd]
            leg_labels_with_miscalibration.append(newLabel)
        # Generate legend
        if alphaColorMap == False:
            leg = ax.legend(
                handles=plotObjects,
                labels=leg_labels_with_miscalibration,
                title=legendTitle,
                loc=leg_loc,
            )
            leg._legend_box.align = legendAlign
        else:
            legendHandles = []
            legendHandles.append((fillObjects[0], plotObjects[0]))
            for trendInd in range(1, len(y_std)):
                legendHandles.append(fillObjects[trendInd])
            leg = ax.legend(
                handles=legendHandles,
                labels=leg_labels_with_miscalibration,
                title=legendTitle,
                loc=leg_loc,
            )
            leg._legend_box.align = legendAlign
    # If a legend is being plotted for one calibration curve,
    elif showLegend == True and type(y_std) is not list:
        # No need to show the miscalibration area in the legend, just label the line.
        leg = ax.legend(handles=plotObjects, labels=leg_labels,
                        title=legendTitle, loc=leg_loc)
        leg._legend_box.align = legendAlign
        # Show the miscalibration of the single calibration curve in its own text box
        ax.text(
            x=0.99,
            y=0.001,
            s='Miscalibration area = %.2f' % miscalibration_areas[0],
            verticalalignment='bottom',
            horizontalalignment='right',
            fontsize=10,
        )
    # If no legend is desired for a single calibration curve,
    elif showLegend == False and type(y_std) is not list:
        # Show the miscalibration of the single calibration curve in its own text box
        ax.text(
            x=0.99,
            y=0.001,
            s='Miscalibration area = %.2f' % miscalibration_areas[0],
            verticalalignment='bottom',
            horizontalalignment='right',
            fontsize=10,
        )

    return ax


def plot_adversarial_group_calibration(
    y_pred: Union[np.ndarray, list],
    y_std: Union[np.ndarray, list],
    y_true: np.ndarray,
    leg_loc: Union[int, str] = 4,
    leg_labels: Union[list, None] = None,
    colorList: list = [
        '#1f77b4',
        'darkorange',
        'darkgreen',
        '#1f77b4',
        'darkorange',
        'darkgreen',
    ],
    markerList: list = ['o', 's', 'p', '^', 'D', 'P'],
    lineList: list = [
        'dashed',
        'dotted',
        'dashdot',
        'solid',
        (0, (3, 1, 1, 1, 1, 1)),
        (0, (3, 5, 1, 5, 1, 5)),
    ],
    legendTitle: Union[str, None] = 'UQ Techniques',
    n_subset: Union[int, None] = None,
    groupSizeLims: Union[tuple, None] = (0.0, 1.0),
    cali_type: str = 'mean_abs',
    curve_label: Union[str, None] = None,
    group_size: Union[np.ndarray, list, None] = None,
    score_mean: Union[np.ndarray, list, None] = None,
    score_stderr: Union[np.ndarray, list, None] = None,
    seed: Union[int, None] = None,
    showTitle: bool = True,
    showLegend: bool = True,
    ax: Union[matplotlib.axes.Axes, None] = None,
) -> matplotlib.axes.Axes:
    """Plot adversarial group calibration plots by varying group size from 0% to 100% of
    dataset size and recording the worst calibration occurred for each group size.

    Args:
        y_pred: 1D array of the predicted means for the held out dataset.
        y_std: 1D array of the predicted standard deviations for the held out dataset.
        y_true: 1D array of the true labels in the held out dataset.
        n_subset: Number of points to plot after filtering.
        cali_type: Calibration type str.
        curve_label: legend label str for calibration curve.
        group_size: 1D array of group size ratios in [0, 1].
        score_mean: 1D array of metric means for group size ratios in group_size.
        score_stderr: 1D array of metric standard devations for group size ratios in group_size.
        ax: matplotlib.axes.Axes object.

    Returns:
        matplotlib.axes.Axes object with plot added.
    """
    # Create ax if it doesn't exist
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    plotObjects = []

    if type(y_pred) is list and type(y_std) is list:

        # Make sure we have enough colors for each trend
        numLineTrends = len(y_std)
        assert len(colorList) >= numLineTrends
        assert len(markerList) >= numLineTrends
        assert len(lineList) >= numLineTrends

        # Optionally select a subset
        if n_subset is not None:
            filteredDatasets = []
            for trendInd in range(len(y_std)):
                filteredDataset = filter_subset(
                    [y_pred[trendInd], y_std[trendInd],
                        y_true[trendInd]], n_subset, seed
                )
                filteredDatasets.append(filteredDataset)

        adv_group_cali_namespaces = []
        group_sizes = []
        score_means = []
        score_stderrs = []
        if (group_size is None) or (score_mean is None):
            for trendInd in range(len(y_std)):
                adv_group_cali_namespace = adversarial_group_calibration(
                    y_pred[trendInd],
                    y_std[trendInd],
                    y_true,
                    cali_type=cali_type,
                    groupSizeLims=groupSizeLims,
                )

                group_size = adv_group_cali_namespace.group_size
                score_mean = adv_group_cali_namespace.score_mean
                score_stderr = adv_group_cali_namespace.score_stderr

                adv_group_cali_namespaces.append(adv_group_cali_namespace)
                group_sizes.append(group_size)
                score_means.append(score_mean)
                score_stderrs.append(score_stderr)

        # If expected and observed proportions are given,
        else:
            for trendInd in range(len(y_std)):
                group_size_iter = np.array(group_size[trendInd]).flatten()
                score_mean_iter = np.array(score_mean[trendInd]).flatten()
                score_stderr_iter = np.array(score_stderr[trendInd]).flatten()

                if (group_size_iter.shape != score_mean_iter.shape) or (
                    group_size_iter.shape != score_stderr.shape
                ):
                    raise RunetimeError(
                        'Input arrays for adversarial group calibration shape mismatch'
                    )

                group_sizes.append(group_size_iter)
                score_means.append(score_mean_iter)
                score_stderrs.append(score_stderr_iter)

        # Plot
        if leg_labels is not None:
            # Make sure that there are enough legend labels for the amount of trends
            assert len(leg_labels) == numLineTrends
            for trendInd in range(len(y_std)):
                (h1,) = ax.plot(
                    group_sizes[trendInd],
                    score_means[trendInd],
                    color='black',
                    marker=markerList[trendInd],
                    markerfacecolor=colorList[trendInd],
                    markeredgecolor='black',
                    markeredgewidth='2',
                    linewidth=2,
                    linestyle=lineList[trendInd],
                    markersize=8,
                    label=leg_labels[trendInd],
                )
                ax.fill_between(
                    group_sizes[trendInd],
                    score_means[trendInd] - score_stderrs[trendInd],
                    score_means[trendInd] + score_stderrs[trendInd],
                    alpha=0.2,
                    color=colorList[trendInd],
                )

                plotObjects.append(h1)
        else:
            for trendInd in range(len(y_std)):
                (h1,) = ax.plot(
                    group_sizes[trendInd],
                    score_means[trendInd],
                    color='black',
                    marker=markerList[trendInd],
                    markerfacecolor=colorList[trendInd],
                    markeredgecolor='black',
                    markeredgewidth='2',
                    linewidth=2,
                    linestyle=lineList[trendInd],
                    markersize=8,
                )
                ax.fill_between(
                    group_sizes[trendInd],
                    score_means[trendInd] - score_stderrs[trendInd],
                    score_means[trendInd] + score_stderrs[trendInd],
                    alpha=0.2,
                    color=colorList[trendInd],
                )

                plotObjects.append(h1)

    else:
        # Optionally select a subset
        if n_subset is not None:
            [y_pred, y_std, y_true] = filter_subset(
                [y_pred, y_std, y_true], n_subset, seed)

        # Compute group_size, score_mean, score_stderr
        if (group_size is None) or (score_mean is None):
            # Compute adversarial group calibration
            adv_group_cali_namespace = adversarial_group_calibration(
                y_pred, y_std, y_true, cali_type=cali_type, groupSizeLims=groupSizeLims
            )
            group_size = adv_group_cali_namespace.group_size
            score_mean = adv_group_cali_namespace.score_mean
            score_stderr = adv_group_cali_namespace.score_stderr
        else:
            # If expected and observed proportions are give
            group_size = np.array(group_size).flatten()
            score_mean = np.array(score_mean).flatten()
            score_stderr = np.array(score_stderr).flatten()
            if (group_size.shape != score_mean.shape) or (
                group_size.shape != score_stderr.shape
            ):
                raise RuntimeError(
                    'Input arrays for adversarial group calibration shape mismatch'
                )

        # Set label
        if curve_label is None:
            curve_label = 'Predictor'

        # Plot
        (h1,) = ax.plot(
            group_size,
            score_mean,
            '-o',
            label=curve_label,
            c='black',
            markerfacecolor='#1f77b4',
            markeredgecolor='black',
            markeredgewidth='2',
            linewidth=2,
            markersize=8,
        )
        ax.fill_between(
            group_size,
            score_mean - score_stderr,
            score_mean + score_stderr,
            alpha=0.2,
            color=colorList[trendInd],
        )

        plotObjects.append(h1)

    # Format plot
    buff = 0.02
    ax.set_xlim([groupSizeLims[0] - buff, groupSizeLims[1] + buff])
    ax.set_ylim([0 - buff, 0.5 + buff])
    ax.set_xlabel('Group Size (Fraction of Test Set)')
    if cali_type == 'miscal_area':
        ax.set_ylabel('Miscalibration Area of Worst Group')
    else:
        ax.set_ylabel('Calibration Error of Worst Group')
    if showTitle == True and type(y_std) is not list:
        ax.set_title('Adversarial Group Calibration (MC Dropout)')
    elif showTitle == True and type(y_std) is list:
        ax.set_title('Adversarial Group Calibration')

    # Legend
    if showLegend == True:
        ax.legend(handles=plotObjects, labels=leg_labels,
                  title=legendTitle, loc=4)

    return ax


def plot_sharpness(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    n_subset: Union[int, None] = None,
    xlims: Union[Tuple[float, float], None] = None,
    bins: Union[int, None] = None,
    seed: Union[int, None] = None,
    showTitle: bool = True,
    ax: Union[matplotlib.axes.Axes, None] = None,
) -> matplotlib.axes.Axes:
    """Plot sharpness of the predictive uncertainties.

    Args:
        y_std: 1D array of the predicted standard deviations for the held out dataset.
        n_subset: Number of points to plot after filtering.
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

    # If bin count is specified, change matlotlib rcParams["hist.bins"] value in histogram,
    if bins is not None:
        # Plot sharpness curve
        ax.hist(y_std, bins=bins, edgecolor='#1f77b4',
                color='#a5c8e1', density=True)
    else:
        # Plot sharpness curve
        ax.hist(y_std, edgecolor='#1f77b4', color='#a5c8e1', density=True)

    # Format plot
    if xlims == None:
        xlims = (y_std.min(), y_std.max())
        ax.set_xlim(xlims)
    else:
        ax.set_xlim(xlims)
    ax.set_xlabel('Adsorption Energy Uncertainty (\u03C3, eV)')
    ax.set_ylabel('Normalized Frequency')

    if showTitle == True:
        ax.set_title('Adsorption Energy Uncertainty (MC Dropout)')
    ax.set_yticklabels([])
    ax.set_yticks([])

    # Calculate and report sharpness
    sharpness = np.sqrt(np.mean(y_std ** 2))
    ax.axvline(x=sharpness, label='sharpness',
               color='#CD0000', linewidth=2, ls='--')

    if sharpness < (xlims[0] + xlims[1]) / 2:
        text = '\n  Sharpness = %.2f eV' % sharpness
        h_align = 'left'
    else:
        text = '\nSharpness = %.2f eV' % sharpness
        h_align = 'right'

    ax.text(
        x=sharpness + 0.05,
        y=ax.get_ylim()[1],
        s=text,
        verticalalignment='top',
        horizontalalignment=h_align,
        fontsize='small',
    )

    ax.text(
        x=ax.get_xlim()[1] - 0.10,
        y=ax.get_ylim()[1] - 0.03,
        s=str(bins) + ' Bins',
        verticalalignment='top',
        horizontalalignment='right',
        fontsize='small',
    )
    return ax


def plot_residuals_vs_stds(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    n_subset: Union[int, None] = None,
    seed: Union[int, None] = None,
    showTitle: bool = True,
    showLegend: bool = True,
    ax: Union[matplotlib.axes.Axes, None] = None,
) -> matplotlib.axes.Axes:
    """Plot absolute value of the prediction residuals versus standard deviations of the
    predictive uncertainties.

    Args:
        y_pred: 1D array of the predicted means for the held out dataset.
        y_std: 1D array of the predicted standard deviations for the held out dataset.
        y_true: 1D array of the true labels in the held out dataset.
        n_subset: Number of points to plot after filtering.
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

    # Compute residuals
    residuals = y_true - y_pred

    # Put stds on same scale as residuals
    residuals_sum = np.sum(np.abs(residuals))
    y_std_scaled = (y_std / np.sum(y_std)) * residuals_sum

    # Plot residuals vs standard devs
    h1 = ax.plot(
        y_std_scaled, np.abs(residuals), '.', c='#1f77b4', mec='black', markeredgewidth=1.0
    )

    # Plot 45-degree line
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    lims = [np.min([xlims[0], ylims[0]]), np.max([xlims[1], ylims[1]])]
    # h2 = ax.plot(lims, lims, "--", c="#CD0000")

    # Identity line
    h2 = ax.plot([0, 1], [0, 1], '--', transform=ax.transAxes,
                 c='#CD0000', linewidth=2)

    # Legend
    if showLegend == True:
        ax.legend([h2[0]], ['y = x'], title=str(n_subset) + ' Clusters', loc=4)

    # Format plot
    ax.set_xlabel('Adsorption Energy Uncertainty\n(Scaled)')
    ax.set_ylabel('Residual Error\n(Absolute, eV)')

    if showTitle == True:
        ax.set_title('Residuals vs. Predictive Uncertainty\n(MC Dropout)')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.axis('square')

    return ax


def plot_residuals_vs_stds_hexagonal(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    gridSize: int = 100,
    bins: Union[str, None] = None,
    subFig: Union[matplotlib.pyplot.figure, None] = None,
    n_subset: Union[int, None] = None,
    ylims: Union[Tuple[float, float], None] = None,
    xlims: Union[Tuple[float, float], None] = None,
    leg_loc: Union[int, str] = 4,
    seed: Union[int, None] = None,
    colorRange: Union[list, tuple, None] = None,
    showTitle: bool = True,
    showLegend: bool = True,
    showColorBar: bool = True,
    getCounts: bool = True,
    ax: Union[matplotlib.axes.Axes, None] = None,
) -> matplotlib.axes.Axes:
    """Plot absolute value of the prediction residuals versus standard deviations of the
    predictive uncertainties, using a hexagonal bin plot to show the data.

    Args:
        y_pred: 1D array of the predicted means for the held out dataset.
        y_std: 1D array of the predicted standard deviations for the held out dataset.
        y_true: 1D array of the true labels in the held out dataset.
        gridSize: Amount of hexagons on the x-axis to plot. Usually, this is auto-matched on the y-axis or some compatible arrangement is automatically determined.
        bins: Whether to calculate hexgonal bin counts based on count, log scale, etc.
        subFig: To plot a color bar with the hexagonal bin, we need the figure object of our plot. Pass this here.
        n_subset: Number of points to plot after filtering
        ylims: a tuple of y axis plotting bounds, given as (lower, upper).
        xlims: a tuple of x axis plotting bounds, given as (lower, upper).
        leg_loc: location of legend as a str or legend code int.
        ax: matplotlib.axes.Axes object

        Returns:
            matplotlib.axes.Axes object with plot added.

    """
    # Create ax if it doesn't exist
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # Color range for heatmaps and color bars is specified as a list or tuple of size 2, (vmin, vmax)
    # Ensure that the list or tuple is of size 2 is an explicit color mapping is specified.
    if colorRange != None:
        assert len(colorRange) == 2

    # Optionally select a subset
    if n_subset is not None:
        [y_pred, y_std, y_true] = filter_subset(
            [y_pred, y_std, y_true], n_subset, seed)

    # Compute residuals
    residuals = y_true - y_pred
    residualsAbs = np.abs(residuals)

    # Put stds on same scale as residuals
    residuals_sum = np.sum(residualsAbs)
    y_std_scaled = (y_std / np.sum(y_std)) * residuals_sum

    if ylims is None:
        lower_upper_ylims = [residualsAbs, y_std_scaled]
        lims_ext_ylims = [
            int(np.floor(np.min(lower_upper_ylims))),
            int(np.ceil(np.max(lower_upper_ylims))),
        ]
    else:
        lims_ext_ylims = ylims
    if xlims is None:
        lower_upper_xlims = [residualsAbs, y_std_scaled]
        lims_ext_xlims = [
            int(np.floor(np.min(lower_upper_xlims))),
            int(np.ceil(np.max(lower_upper_xlims))),
        ]
    else:
        lims_ext_xlims = xlims

    # Identity line
    h2 = ax.plot([0, 1], [0, 1], '--', transform=ax.transAxes,
                 c='#CD0000', linewidth=2)

    if colorRange != None:
        h1 = ax.hexbin(
            y_std_scaled,
            np.abs(residuals),
            gridsize=gridSize,
            bins=bins,
            extent=(
                lims_ext_xlims[0],
                lims_ext_xlims[1],
                lims_ext_ylims[0],
                lims_ext_ylims[1],
            ),
            cmap='Blues',
            vmin=colorRange[0],
            vmax=colorRange[1],
            linewidths=0.1,
        )
    else:
        h1 = ax.hexbin(
            y_std_scaled,
            np.abs(residuals),
            gridsize=gridSize,
            bins=bins,
            extent=(
                lims_ext_xlims[0],
                lims_ext_xlims[1],
                lims_ext_ylims[0],
                lims_ext_ylims[1],
            ),
            cmap='Blues',
            linewidths=0.1,
        )

    counts = h1.get_array()

    if showColorBar == True:
        cb = subFig.colorbar(h1, ax=ax, pad=0.01)
        if bins == None:
            cb.set_label('Samples')
        elif bins == 'log':
            cb.set_label('log10(Samples)')

    if n_subset is not None:
        titleName = str(n_subset) + ' Clusters'
    else:
        titleName = None

    # Legend
    if showLegend == True:
        ax.legend([h2[0]], ['Perfect Fit'], title=titleName, loc=leg_loc)

    # Format plot
    ax.set_xlabel('Adsorption Energy Uncertainty\n(Scaled)')
    ax.set_ylabel('Residual Error\n(Absolute, eV)')

    if showTitle == True:
        ax.set_title('Residuals vs. Predictive Uncertainty\n(MC Dropout)')
    ax.set_xlim(lims_ext_xlims)
    ax.set_ylim(lims_ext_ylims)

    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')
    # ax.axis("square")

    # If we want to count the minimum or maximum counts in the hexagons,
    if getCounts == True:
        # Store the plot and hexagon counts into a tuple and return it.
        axTuple = (ax, counts)
        return axTuple
    else:
        return ax


def plot_violin_box_plot(
    y_std: Union[np.ndarray, list],
    ylims: Union[Tuple[float, float], None] = None,
    leg_loc: Union[int, str] = 4,
    seed: Union[int, None] = None,
    kdePoints: Union[int] = 100,
    width: float = 0.5,
    whiskers: Union[tuple, None] = None,
    aspectRatio: Union[float, None] = None,
    showTitle: bool = True,
    showLegend: bool = True,
    showFliers: bool = True,
    showMeans: bool = False,
    showExtrema: bool = False,
    showMedians: bool = False,
    showVertical: bool = True,
    statLabels: Union[list, None] = None,
    ax: Union[matplotlib.axes.Axes, None] = None,
) -> matplotlib.axes.Axes:
    # Create ax if it doesn't exist
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # Color range for heatmaps and color bars is specified as a list or tuple of size 2, (vmin, vmax)
    # Ensure that the list or tuple is of size 2 is an explicit color mapping is specified.
    # if colorRange != None:
    #    assert len(colorRange) == 2

    if type(y_std) is list:

        if ylims is None:
            lower_upper_ylims = y_std
            minCandidates = []
            maxCandidates = []

            for dataset in y_std:
                minCandidate = np.min(dataset)
                maxCandidate = np.max(dataset)

                minCandidates.append(minCandidate)
                maxCandidates.append(maxCandidate)

            plotMin = np.min(minCandidates)
            plotMax = np.max(maxCandidates)

            lims_ext_ylims = [
                plotMin,
                plotMax,
            ]

            # Move the plot slightly upwards by increasing the plotting range to prevent the violin/box plot from being obscured by the figure border.
            percentToBump = 0.05
            amountToBump = (lims_ext_ylims[1] -
                            lims_ext_ylims[0]) * percentToBump
            newMin = lims_ext_ylims[0] - amountToBump
            newMax = lims_ext_ylims[1] + amountToBump

            lims_ext_ylims[0] = newMin
            lims_ext_ylims[1] = newMax
        else:
            lims_ext_ylims = ylims
    else:

        if ylims is None:
            lower_upper_ylims = y_std
            lims_ext_ylims = [
                np.min(lower_upper_ylims),
                np.max(lower_upper_ylims),
            ]

            # Move the plot slightly upwards by increasing the plotting range to prevent the violin/box plot from being obscured by the figure border.
            percentToBump = 0.05
            amountToBump = (lims_ext_ylims[1] -
                            lims_ext_ylims[0]) * percentToBump
            newMin = lims_ext_ylims[0] - amountToBump
            newMax = lims_ext_ylims[1] + amountToBump

            lims_ext_ylims[0] = newMin
            lims_ext_ylims[1] = newMax
        else:
            lims_ext_ylims = ylims

    # Violin plot
    h1 = ax.violinplot(
        y_std,
        vert=showVertical,
        widths=width,
        showmeans=showMeans,
        showextrema=showExtrema,
        showmedians=showMeans,
        points=kdePoints,
    )
    # Violin plot is a polycollection, so access the polycollection elements and use polycollection features to change the color
    for violinPart in h1['bodies']:
        violinPart.set_facecolor('steelblue')
        violinPart.set_linewidth(2)
        violinPart.set_edgecolor('midnightblue')
        violinPart.set_alpha(0.45)

    # Box plot
    h2 = ax.boxplot(
        y_std,
        vert=showVertical,
        widths=width / 5,
        whis=whiskers,
        showmeans=True,
        meanline=True,
        showfliers=showFliers,
        boxprops={'linewidth': 1.5},
        whiskerprops={'linewidth': 1.5},
        capprops={'linewidth': 1.5},
        medianprops={'linewidth': 1.5, 'color': 'black'},
        meanprops={'linestyle': (0, (1, 1)),
                   'linewidth': 1.5, 'color': '#CD0000'},
        flierprops={'marker': '_', 'markeredgewidth': 0.25},
    )

    # if colorRange != None:
    #    h1 = ax.hexbin(y_std_scaled, np.abs(residuals), gridsize = gridSize, bins=bins, extent = (lims_ext_xlims[0], lims_ext_xlims[1], lims_ext_ylims[0], lims_ext_ylims[1]), cmap = "Blues",
    #        vmin=colorRange[0], vmax=colorRange[1])
    # else:
    #    h1 = ax.hexbin(y_std_scaled, np.abs(residuals), gridsize = gridSize, bins=bins, extent = (lims_ext_xlims[0], lims_ext_xlims[1], lims_ext_ylims[0], lims_ext_ylims[1]), cmap = "Blues")

    # Legend
    # if showLegend == True:
    #    ax.legend([h2[0]], ["Perfect Fit"], title = titleName, loc=leg_loc)

    # Format plot
    ax.set_ylabel('Adsorption Energy Uncertainty\n(\u03C3, eV)')
    ax.set_xlabel('5-fold Ensemble')
    ax.set_xticklabels([])

    if showTitle == True:
        ax.set_title(
            'Adsorption Energy Uncertainty Violin Plot\n(5-fold Ensemble)')
    # ax.set_xlim(lims_ext_xlims)
    ax.set_ylim(lims_ext_ylims)

    if aspectRatio != None:
        ax.set_aspect(aspectRatio / ax.get_data_ratio(), adjustable='box')
    else:
        ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')

    # If we are labelling the sharpness,
    if statLabels is not None:
        # Get the xLims of each box plot
        box_xLims = []
        for element in ['medians']:
            for lineInd in range(len(h2[element])):
                # Returns data coordinates
                (xLeft, y), (xRight, _) = h2[element][lineInd].get_xydata()
                box_xLims.append(list(zip([xLeft], [xRight])))

        # Plot the sharpness line and label it
        for statInd in range(len(statLabels)):
            ax.plot(
                [box_xLims[statInd][0][0], box_xLims[statInd][0][1]],
                [statLabels[statInd][0][1], statLabels[statInd][0][1]],
                color='#CD0000',
                linewidth=1.5,
                linestyle=(0, (1, 1)),
            )
            labelXCenter = (
                box_xLims[statInd][0][1]
                + (box_xLims[statInd][0][1] - box_xLims[statInd][0][0]) / 8
            )
            labelYCenter = statLabels[statInd][0][1]
            subStr = (
                statLabels[statInd][0][0]
                + ' = '
                + '{:<6.3f}'.format(statLabels[statInd][0][1])
            )
            ax.text(
                labelXCenter,
                labelYCenter,
                subStr,
                fontsize=8,
                verticalalignment='center',
                horizontalalignment='left',
                color='#CD0000',
                bbox=dict(
                    edgecolor='black', facecolor='white', linewidth=1.0, alpha=1.0, pad=0.75
                ),
                weight='bold',
            )

    return ax


def filter_subset(
    input_list: List[List[Any]], n_subset: int, seed: Union[int, None] = None
) -> List[List[Any]]:
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


def set_style(style_str: str = 'default') -> NoReturn:
    """Set the matplotlib plotting style.

    Args:
        style_str: string for style file.
    """
    if style_str == 'default':
        plt.style.use(
            (pathlib.Path(__file__).parent / 'matplotlibrc').resolve())


def save_figure(
    file_name: str = 'figure',
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
        ext_list = ['pdf', 'png']

    # If ext_list is a single str
    if isinstance(ext_list, str):
        ext_list = [ext_list]

    # Set facecolor and edgecolor
    (fc, ec) = ('w', 'w') if white_background else ('none', 'none')

    # Save each type in ext_list
    for ext in ext_list:
        save_str = file_name + '.' + ext
        plt.savefig(save_str, bbox_inches='tight', facecolor=fc, edgecolor=ec)
        print(f'Saved figure {save_str}')


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
