import uncertainty_toolbox as uct
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import os
import os.path

# Save the unicode string for the lambda symbol so we can put it in the plots later.
# Make this a global variable such that it is accessible from all functions.
global lambdaSymbolStr
lambdaSymbolStr = "\u03bb"


def make_plots(pred_mean, pred_std, target, lamb, subsetSeed, lineplot_array=None, lineplot_legLabels=None, lineplot_xLabels=None, showTitleChoice=True, showLegendChoice=True, plot_save_str="row"):
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
    uct.viz_Evidential.update_rc("legend.fontsize", 6.5)
    # Configure legend title font size
    uct.viz_Evidential.update_rc("legend.title_fontsize", 6.5)
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

    fig, axs = plt.subplots(1, 1, figsize=(3.8, 3.8),
                            sharex=False, sharey=False)

    print(axs)

    # Make calibration plot
    epistemicLabel = "Epistemic Evidential"
    aleatoricLabel = "Aleatoric Evidential"
    axs = uct.viz_MC_Dropout.plot_calibration(pred_mean, pred_std, target, leg_loc=2, leg_labels=["5-fold Ensemble", "MC Dropout", epistemicLabel, aleatoricLabel], seed=subsetSeed, showTitle=showTitleChoice, showLegend=showLegendChoice, showAlpha=True,
                                              alphaColorMap=True, ax=axs)
    axs.yaxis.set_ticks(np.arange(0.0, 1.2, 0.2))
    axs.xaxis.set_ticks(np.arange(0.0, 1.2, 0.2))
    if showTitleChoice == True:
        axs.set(title="Average Calibration\nAcross UQ Techniques")

    # Adjust subplots spacing
    # fig.subplots_adjust(wspace=0.75)
    # fig.subplots_adjust(hspace=0.50)
    plt.tight_layout()
    axs.grid(visible=None)
    axs.tick_params(axis="both", which="major", direction="out",
                    color="black", length=4.0, width=2.0)
    # minorTicksLocator = AutoMinorLocator(2)
    # axs.xaxis.set_minor_locator(minorTicksLocator)
    # axs.yaxis.set_minor_locator(minorTicksLocator)
    # axs.tick_params(axis="both", which="minor", direction="out", color="black", length=3.0, width=1.25)
    # Save figure
    if savefig:
        uct.viz_Evidential.save_figure(
            plot_save_str, "svg", white_background=True)
        uct.viz_Evidential.save_figure(
            plot_save_str, "png", white_background=True)


targetFilePath = "VALID_TARGETS_FILEPATH"

predictionFilePathList = []
predictionFilePathEnsemble = "ENSEMBLE_PREDICTIONS_FILEPATH"
predictionFilePathDropout = "MC_DROPOUT_PREDICTIONS_FILEPATH"
predictionFilePathEvidential = "EVIDENTIAL_PREDICTIONS_FILEPATH"
predictionFilePathList.append(predictionFilePathEnsemble)
predictionFilePathList.append(predictionFilePathDropout)
predictionFilePathList.append(predictionFilePathEvidential)
predictionFilePathList.append(predictionFilePathEvidential)

uncertaintyFilePathList = []
uncertaintyFilePathEnsemble = "ENSEMBLE_UNCERTAINTY_FILEPATH"
uncertaintyFilePathDropout = "MC_DROPOUT_UNCERTAINTY_FILEPATH"
uncertaintyFilePathEvidential = "EVIDENTIAL_UNCERTAINTY_FILEPATH"
uncertaintyFilePathList.append(uncertaintyFilePathEnsemble)
uncertaintyFilePathList.append(uncertaintyFilePathDropout)
uncertaintyFilePathList.append(uncertaintyFilePathEvidential)
uncertaintyFilePathList.append(uncertaintyFilePathEvidential)

print(predictionFilePathList)

# Where to save plot
savePlotPath = "CALIBRATION_OVERLAY_PLOT_SAVE_FILEPATH"

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
predictionFileNameList.append(predictionFileNameEvidential)


uncertaintyFileNameList = []
uncertaintyFileNameEnsemble = "IS2RE_all_CGCNN_ValID_Ensemble5_Energy_StdDev.csv"
uncertaintyFileNameDropout = "IS2RE_all_CGCNN_ValID_MC_Energy_StdDev1000.csv"
epistemicUncertaintyFileNameEvidential = "epistemic_uncertainty_" + jobID + ".csv"
aleatoricUncertaintyFileNameEvidential = "aleatoric_uncertainty_" + jobID + ".csv"
uncertaintyFileNameList.append(uncertaintyFileNameEnsemble)
uncertaintyFileNameList.append(uncertaintyFileNameDropout)
uncertaintyFileNameList.append(epistemicUncertaintyFileNameEvidential)
uncertaintyFileNameList.append(aleatoricUncertaintyFileNameEvidential)

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
savePlotNameTitle = "UQ_Technique_Compare_Overlay_Calibration_Title_Seed" + \
    str(rngVisualSeed)
savePlotNameLeg = "UQ_Technique_Compare_Overlay_Calibration_Leg_Seed" + \
    str(rngVisualSeed)
savePlotNameLegTitle = "UQ_Technique_Compare_Overlay_Calibration_LegTitle_Seed" + \
    str(rngVisualSeed)
savePlotName = "UQ_Technique_Compare_Overlay_Calibration_Seed" + \
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
aleatoricUncertaintyColNameEvidential = "Aleatoric Uncertainty (std. dev., eV)"
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
predictionNPList.append(predictionNPEvidential)

uncertaintyNPEnsemble = uncertaintyPDList[0][uncertaintyColNameEnsemble].to_numpy(
)
uncertaintyNPDropout = uncertaintyPDList[1][uncertaintyColNameDropout].to_numpy(
)
epistemicUncertaintyNPEvidential = uncertaintyPDList[2][epistemicUncertaintyColNameEvidential].to_numpy(
)
aleatoricUncertaintyNPEvidential = uncertaintyPDList[3][aleatoricUncertaintyColNameEvidential].to_numpy(
)
uncertaintyNPList.append(uncertaintyNPEnsemble)
uncertaintyNPList.append(uncertaintyNPDropout)
uncertaintyNPList.append(epistemicUncertaintyNPEvidential)
uncertaintyNPList.append(aleatoricUncertaintyNPEvidential)

# Save figure specification
savefig = True

# Whether the title and/or legend is going to be shown.
make_plots(predictionNPList, uncertaintyNPList, targetNP, lamb, rngVisualSeed, lineplot_array=None, lineplot_legLabels=None, lineplot_xLabels=None,
           showTitleChoice=True, showLegendChoice=False, plot_save_str=savePlotFileTitle)
make_plots(predictionNPList, uncertaintyNPList, targetNP, lamb, rngVisualSeed, lineplot_array=None, lineplot_legLabels=None, lineplot_xLabels=None,
           showTitleChoice=False, showLegendChoice=True, plot_save_str=savePlotFileLeg)
make_plots(predictionNPList, uncertaintyNPList, targetNP, lamb, rngVisualSeed, lineplot_array=None, lineplot_legLabels=None, lineplot_xLabels=None,
           showTitleChoice=True, showLegendChoice=True, plot_save_str=savePlotFileLegTitle)
make_plots(predictionNPList, uncertaintyNPList, targetNP, lamb, rngVisualSeed, lineplot_array=None, lineplot_legLabels=None, lineplot_xLabels=None,
           showTitleChoice=False, showLegendChoice=False, plot_save_str=savePlotFile)
