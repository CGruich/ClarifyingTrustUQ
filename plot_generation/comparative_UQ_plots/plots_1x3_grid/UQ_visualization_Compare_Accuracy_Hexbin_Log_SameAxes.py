import uncertainty_toolbox as uct
import pandas as pd
import numpy as np
# For easily adding color bars to grids of images
from mpl_toolkits.axes_grid1 import make_axes_locatable
# For easily defining color bars
import matplotlib as mpl
# For plotting in general
import matplotlib.pyplot as plt
import os
import os.path

# Save the unicode string for the lambda symbol so we can put it in the plots later.
# Make this a global variable such that it is accessible from all functions.
global lambdaSymbolStr
lambdaSymbolStr = "\u03bb"


def make_plots(pred_mean, pred_std, target, lamb, subsetSeed, lineplot_array=None, lineplot_legLabels=None, lineplot_xLabels=None, accuracyStats=None, showTitleChoice=True, showLegendChoice=True, plot_save_str="row"):
    """Make set of plots."""

    # "subsetSeed" is for plotting reproducibility any time data is binned (i.e. collected and averaged) for the purposes of visualization.
    # Turn on plot style customization
    uct.viz_Evidential.set_style()
    # Change font size in general
    uct.viz_Evidential.update_rc("font.size", 14)  # Set font size
    # Change font fize of the title
    # uct.viz_Evidential.update_rc("figure.titlesize", 16) # Set title font size
    # Change xtick label size
    # Set font size for xaxis tick labels
    uct.viz_Evidential.update_rc("xtick.labelsize", 14)
    # Change ytick label size
    # Set font size for yaxis tick labels
    uct.viz_Evidential.update_rc("ytick.labelsize", 14)
    # Turn off Latex mode
    uct.viz_Evidential.update_rc("text.usetex", False)
    # Update legend general font size
    uct.viz_Evidential.update_rc("legend.fontsize", 10)
    # Configure legend title font size
    uct.viz_Evidential.update_rc("legend.title_fontsize", "small")
    # Poster quality plots
    uct.viz_Evidential.update_rc("figure.dpi", 600)
    # Change the font globally
    uct.viz_Evidential.update_rc("font.family", "Times New Roman")
    # Change the figure border width globally
    uct.viz_Evidential.update_rc("axes.linewidth", 2.0)
    # Change the figure border color globally
    uct.viz_Evidential.update_rc("axes.edgecolor", "black")

    # ylims = [-20, 20]

    subsetCountOrderedIntervals = 50
    subsetCountSharpness = None
    subsetCountCalibration = None
    subsetCountAdversarialCalibration = None
    subsetCountResidualUncertainty = 200
    subsetCountParity = 200

    binCount = 100

    fig, axs = plt.subplots(1, 3, figsize=(9, 40), sharex=False, sharey=False)

    print(axs)

    gridSizeCount = 40

    print("\nProducing plots that are not color scaled to same range...")
    # Make a parity plot, but with hexagonal binning. This shows model accuracy on a sample-by-sample basis.
    axs[0], countsEnsemble = uct.viz_Ensemble.plot_parity_hexagonal(pred_mean[0], pred_std[0], target, gridSize=gridSizeCount, bins="log", subFig=fig, n_subset=None,
                                                                    leg_loc=4, colorRange=None, seed=subsetSeed, showTitle=showTitleChoice, showLegend=showLegendChoice, showColorBar=False, getCounts=True, statLabels=accuracyStats[0], ax=axs[0])
    axs[0].set(xlabel=None)
    if showTitleChoice == True:
        axs[0].set(title="5-fold Ensemble")
    axs[1], countsDropout = uct.viz_MC_Dropout.plot_parity_hexagonal(pred_mean[1], pred_std[1], target, gridSize=gridSizeCount, bins="log", subFig=fig, n_subset=None,
                                                                     leg_loc=4, colorRange=None, seed=subsetSeed, showTitle=showTitleChoice, showLegend=showLegendChoice, showColorBar=False, getCounts=True, statLabels=accuracyStats[1], ax=axs[1])
    axs[1].set(ylabel=None)
    if showTitleChoice == True:
        axs[1].set(title="MC Dropout")
    axs[2], countsEvidential = uct.viz_Evidential.plot_parity_hexagonal(pred_mean[2], pred_std[2], target, lamb, gridSize=gridSizeCount, bins="log", subFig=fig, n_subset=None,
                                                                        leg_loc=4, colorRange=None, seed=subsetSeed, showTitle=showTitleChoice, showLegend=showLegendChoice, showColorBar=False, getCounts=True, statLabels=accuracyStats[2], ax=axs[2])
    axs[2].set(xlabel=None)
    axs[2].set(ylabel=None)
    if showTitleChoice == True:
        axs[2].set(title="Evidential Regression")

    commonTickLims = uct.viz_Ensemble.determine_common_axes(axs)
    plotLimBuffer = 0.1
    commonPlotLims = ((commonTickLims[0][0] - plotLimBuffer, commonTickLims[0][1] + plotLimBuffer),
                      (commonTickLims[1][0] - plotLimBuffer, commonTickLims[1][1] + plotLimBuffer))

    countType = (countsEnsemble, countsDropout, countsEvidential)
    colorBarLims = uct.viz_Evidential.determine_common_colorbar(countType)

    # By convention, we need to add +1 to the color bar limits when trying to make a log-scale color bar
    # This is because log(0) is undefined
    colorBarLimsLogCompatible = list(colorBarLims)
    colorBarLimsLogCompatible[0] = colorBarLimsLogCompatible[0] + 1
    colorBarLims = colorBarLimsLogCompatible

    print("\nReproducing plots that are on the same color scale, as well as adding color bar...")
    print("Common Color Bar Range Chosen: ")
    print(colorBarLims)

    axs[0], countsEnsemble = uct.viz_Ensemble.plot_parity_hexagonal(pred_mean[0], pred_std[0], target, gridSize=gridSizeCount, bins="log", subFig=fig, n_subset=None,
                                                                    ylims=commonPlotLims[0], xlims=commonPlotLims[1], leg_loc=4, colorRange=colorBarLims, seed=subsetSeed,
                                                                    showTitle=showTitleChoice, showLegend=showLegendChoice, showColorBar=False, getCounts=True, ax=axs[0])
    axs[0].grid(visible=None)
    axs[0].set(xlabel=None)
    xlim = axs[0].get_xlim()
    ylim = axs[0].get_ylim()
    axs[0].xaxis.set_ticks(
        np.arange(commonTickLims[0][0], commonTickLims[0][1] + 1, 4.0))
    axs[0].yaxis.set_ticks(
        np.arange(commonTickLims[0][0], commonTickLims[0][1] + 1, 4.0))
    axs[0].tick_params(axis="both", which="major",
                       direction="out", color="black", length=4.0, width=2.0)
    if showTitleChoice == True:
        axs[0].set(title="5-fold Ensemble")

    axs[1], countsDropout = uct.viz_MC_Dropout.plot_parity_hexagonal(pred_mean[1], pred_std[1], target, gridSize=gridSizeCount, bins="log", subFig=fig, n_subset=None,
                                                                     ylims=commonPlotLims[0], xlims=commonPlotLims[1], leg_loc=4, colorRange=colorBarLims, seed=subsetSeed, showTitle=showTitleChoice, showLegend=showLegendChoice, showColorBar=False, getCounts=True, ax=axs[1])
    axs[1].grid(visible=None)
    axs[1].set(ylabel=None)
    xlim = axs[1].get_xlim()
    ylim = axs[1].get_ylim()
    axs[1].xaxis.set_ticks(
        np.arange(commonTickLims[0][0], commonTickLims[0][1] + 1, 4.0))
    axs[1].yaxis.set_ticks(
        np.arange(commonTickLims[0][0], commonTickLims[0][1] + 1, 4.0))
    axs[1].tick_params(axis="both", which="major",
                       direction="out", color="black", length=4.0, width=2.0)
    if showTitleChoice == True:
        axs[1].set(title="MC Dropout")

    axs[2], countsEvidential = uct.viz_Evidential.plot_parity_hexagonal(pred_mean[2], pred_std[2], target, lamb, gridSize=gridSizeCount, bins="log", subFig=fig, n_subset=None,
                                                                        ylims=commonPlotLims[0], xlims=commonPlotLims[1], leg_loc=4, colorRange=colorBarLims, seed=subsetSeed, showTitle=showTitleChoice, showLegend=showLegendChoice, showColorBar=False, getCounts=True, ax=axs[2])
    axs[2].grid(visible=None)
    axs[2].set(xlabel=None)
    axs[2].set(ylabel=None)
    xlim = axs[2].get_xlim()
    ylim = axs[2].get_ylim()
    axs[2].xaxis.set_ticks(
        np.arange(commonTickLims[0][0], commonTickLims[0][1] + 1, 4.0))
    axs[2].yaxis.set_ticks(
        np.arange(commonTickLims[0][0], commonTickLims[0][1] + 1, 4.0))
    axs[2].tick_params(axis="both", which="major",
                       direction="out", color="black", length=4.0, width=2.0)
    if showTitleChoice == True:
        axs[2].set(title="Evidential Regression")

    plt.tight_layout()
    # colorBarDivider = make_axes_locatable(axs[2])
    scalarMap = plt.cm.ScalarMappable(cmap="Blues", norm=mpl.colors.LogNorm(
        vmin=colorBarLims[0], vmax=colorBarLims[1]))
    rightMostPlotPosition = axs[2].get_position()
    # Returns boundary box point coordinates in the form of [(x0, y0), (x1, y1)]
    rightMostPlotCorners = rightMostPlotPosition.get_points()
    padding = 0.01
    print("Position of BBox to attach colorbar to:")
    print(rightMostPlotCorners)
    colorBarAx = plt.axes([rightMostPlotCorners[1][0] + padding,
                          rightMostPlotCorners[0][1], 0.0125, rightMostPlotPosition.height])
    # colorBarAx = colorBarDivider.append_axes("right", size = "5%", pad=0.05)
    colorBar = plt.colorbar(scalarMap, colorBarAx)
    # colorBar.set_label("Samples")
    colorBar.set_label("log10(Samples)")

    # Adjust subplots spacing
    fig.subplots_adjust(wspace=0.22)
    # fig.subplots_adjust(hspace=0.50)

    # Save figure
    if savefig:
        uct.viz_Evidential.save_figure(
            plot_save_str, "svg", white_background=True)
        uct.viz_Evidential.save_figure(
            plot_save_str, "png", white_background=True)


targetFilePath = "VALID_TARGETS_FILEPATH"

predictionFilePathList = []
predictionFilePathEnsemble = "ENSEMBLE_PREDICTION_FILEPATH"
predictionFilePathDropout = "DROPOUT_PREDICTION_FILEPATH"
predictionFilePathEvidential = "EVIDENTIAL_PREDICTION_FILEPATH"
predictionFilePathList.append(predictionFilePathEnsemble)
predictionFilePathList.append(predictionFilePathDropout)
predictionFilePathList.append(predictionFilePathEvidential)

uncertaintyFilePathList = []
uncertaintyFilePathEnsemble = "ENSEMBLE_UNCERTAINTY_FILEPATH"
uncertaintyFilePathDropout = "DROPOUT_UNCERTAINTY_FILEPATH"
uncertaintyFilePathEvidential = "EVIDENTIAL_UNCERTAINTY_FILEPATH"
uncertaintyFilePathList.append(uncertaintyFilePathEnsemble)
uncertaintyFilePathList.append(uncertaintyFilePathDropout)
uncertaintyFilePathList.append(uncertaintyFilePathEvidential)

print(predictionFilePathList)

# Where to save plot
savePlotPath = "SAVE_PLOT_PATH_USER_SPECIFIED"

# Create directory to save the plot to, if it does not exist
if not os.path.exists(savePlotPath):
    os.makedirs(savePlotPath)

# Random number generation seed for visualization
rngVisualSeed = 100

# Lambda hyperparameter used for DER
lamb = 0.05
jobID = "9600051"
lambStr = str(lamb).replace(".", "p")

targetFileName = "targets.csv"

predictionFileNameList = []
predictionFileNameEnsemble = "IS2RE_all_CGCNN_ValID_Ensemble5_Energy_Mean.csv"
predictionFileNameDropout = "IS2RE_all_CGCNN_ValID_MC_Energy_Mean1000.csv"
predictionFileNameEvidential = "evidential_gamma_" + jobID + ".csv"
predictionFileNameList.append(predictionFileNameEnsemble)
predictionFileNameList.append(predictionFileNameDropout)
predictionFileNameList.append(predictionFileNameEvidential)


uncertaintyFileNameList = []
uncertaintyFileNameEnsemble = "IS2RE_all_CGCNN_ValID_Ensemble5_Energy_StdDev.csv"
uncertaintyFileNameDropout = "IS2RE_all_CGCNN_ValID_MC_Energy_StdDev1000.csv"
epistemicUncertaintyFileNameEvidential = "epistemic_uncertainty_" + jobID + ".csv"
uncertaintyFileNameList.append(uncertaintyFileNameEnsemble)
uncertaintyFileNameList.append(uncertaintyFileNameDropout)
uncertaintyFileNameList.append(epistemicUncertaintyFileNameEvidential)

print(uncertaintyFileNameList)

predictionFileList = []
uncertaintyFileList = []

for ind in range(len(predictionFilePathList)):
    predictionFilePath = os.path.join(
        predictionFilePathList[ind], predictionFileNameList[ind])
    uncertaintyFilePath = os.path.join(
        uncertaintyFilePathList[ind], uncertaintyFileNameList[ind])

    predictionFileList.append(predictionFilePath)
    uncertaintyFileList.append(uncertaintyFilePath)


print("\nPrediction Files:")
print(predictionFileList)
print("\nUncertainty Files:")
print(uncertaintyFileList)


targetFile = targetFilePath + targetFileName
# What to name plot
# aleatoricSavePlotName = "Aleatoric_Lamb" + lambStr + "_Seed" + str(rngVisualSeed)
savePlotNameTitle = "UQ_Technique_Compare_Accuracy_Hexbin_Log_Title_SameAxes_Seed" + \
    str(rngVisualSeed)
savePlotNameLeg = "UQ_Technique_Compare_Accuracy_Hexbin_Log_Leg_SameAxes_Seed" + \
    str(rngVisualSeed)
savePlotNameLegTitle = "UQ_Technique_Accuracy_Hexbin_Log_LegTitle_SameAxes_Seed" + \
    str(rngVisualSeed)
savePlotName = "UQ_Technique_Compare_Accuracy_Hexbin_Log_SameAxes_Seed" + \
    str(rngVisualSeed)

savePlotFileTitle = os.path.join(savePlotPath, savePlotNameTitle)
savePlotFileLeg = os.path.join(savePlotPath, savePlotNameLeg)
savePlotFileLegTitle = os.path.join(savePlotPath, savePlotNameLegTitle)
savePlotFile = os.path.join(savePlotPath, savePlotName)

targetPD = pd.read_csv(targetFile)


predictionPDList = []
uncertaintyPDList = []

for ind in range(len(predictionFileList)):
    predictionPD = pd.read_csv(predictionFileList[ind])
    predictionPDList.append(predictionPD)
    uncertaintyPD = pd.read_csv(uncertaintyFileList[ind])
    uncertaintyPDList.append(uncertaintyPD)

targetColName = "Target (eV)"
predictionColNameEnsemble = "Mean Sample Energy (eV)"
predictionColNameDropout = predictionColNameEnsemble
predictionColNameEvidential = "Ads. Energy. (eV)"
uncertaintyColNameEnsemble = "Std. Dev. Sample Energy (eV)"
uncertaintyColNameDropout = uncertaintyColNameEnsemble
epistemicUncertaintyColNameEvidential = "Epistemic Uncertainty (std. dev., eV)"

targetNP = targetPD[targetColName].to_numpy()

predictionNPList = []
uncertaintyNPList = []

predictionNPEnsemble = predictionPDList[0][predictionColNameEnsemble].to_numpy(
)
predictionNPDropout = predictionPDList[1][predictionColNameDropout].to_numpy()
predictionNPEvidential = predictionPDList[2][predictionColNameEvidential].to_numpy(
)
predictionNPList.append(predictionNPEnsemble)
predictionNPList.append(predictionNPDropout)
predictionNPList.append(predictionNPEvidential)

uncertaintyNPEnsemble = uncertaintyPDList[0][uncertaintyColNameEnsemble].to_numpy(
)
uncertaintyNPDropout = uncertaintyPDList[1][uncertaintyColNameDropout].to_numpy(
)
epistemicUncertaintyNPEvidential = uncertaintyPDList[2][epistemicUncertaintyColNameEvidential].to_numpy(
)
uncertaintyNPList.append(uncertaintyNPEnsemble)
uncertaintyNPList.append(uncertaintyNPDropout)
uncertaintyNPList.append(epistemicUncertaintyNPEvidential)

# Save figure specification
savefig = True

# List of tuples
accuracyStatLabels = ["MAE", "RMSE", "MDAE", "MARPD", "R2", "R"]
ensembleAccuracyStats = [0.671, 1.039, 0.427, 48.194, 0.791, 0.890]
ensembleAccuracyStats = list(zip(accuracyStatLabels, ensembleAccuracyStats))
dropoutAccuracyStats = [0.664, 1.026, 0.425, 47.941, 0.797, 0.893]
dropoutAccuracyStats = list(zip(accuracyStatLabels, dropoutAccuracyStats))
evidentialAccuracyStats = [0.630, 0.985, 0.391, 45.968, 0.812, 0.902]
evidentialAccuracyStats = list(
    zip(accuracyStatLabels, evidentialAccuracyStats))

accuracyStats = []
accuracyStats.append(ensembleAccuracyStats)
accuracyStats.append(dropoutAccuracyStats)
accuracyStats.append(evidentialAccuracyStats)
# Whether the title and/or legend is going to be shown.
make_plots(predictionNPList, uncertaintyNPList, targetNP, lamb, rngVisualSeed, lineplot_array=None, lineplot_legLabels=None, lineplot_xLabels=None,
           accuracyStats=accuracyStats, showTitleChoice=True, showLegendChoice=False, plot_save_str=savePlotFileTitle)
make_plots(predictionNPList, uncertaintyNPList, targetNP, lamb, rngVisualSeed, lineplot_array=None, lineplot_legLabels=None, lineplot_xLabels=None,
           accuracyStats=accuracyStats, showTitleChoice=False, showLegendChoice=True, plot_save_str=savePlotFileLeg)
make_plots(predictionNPList, uncertaintyNPList, targetNP, lamb, rngVisualSeed, lineplot_array=None, lineplot_legLabels=None, lineplot_xLabels=None,
           accuracyStats=accuracyStats, showTitleChoice=True, showLegendChoice=True, plot_save_str=savePlotFileLegTitle)
make_plots(predictionNPList, uncertaintyNPList, targetNP, lamb, rngVisualSeed, lineplot_array=None, lineplot_legLabels=None, lineplot_xLabels=None,
           accuracyStats=accuracyStats, showTitleChoice=False, showLegendChoice=False, plot_save_str=savePlotFile)
