import uncertainty_toolbox as uct
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import PercentFormatter
import os
import os.path

# Save the unicode string for the lambda symbol so we can put it in the plots later.
# Make this a global variable such that it is accessible from all functions.
global lambdaSymbolStr
lambdaSymbolStr = "\u03bb"

def make_plots(pred_mean, pred_std, target, lamb, subsetSeed, lineplot_array = None, lineplot_legLabels = None, lineplot_xLabels = None, showTitleChoice = True, showLegendChoice = True, plot_save_str="row"):
    """Make set of plots."""

    # Turn on plot style customization
    uct.viz_Evidential.set_style()
    # Change font size in general
    uct.viz_Evidential.update_rc("font.size", 14)  # Set font size
    # Change font fize of the title
    #uct.viz_Evidential.update_rc("figure.titlesize", 16) # Set title font size
    # Change xtick label size
    uct.viz_Evidential.update_rc("xtick.labelsize", 14)  # Set font size for xaxis tick labels
    # Change ytick label size
    uct.viz_Evidential.update_rc("ytick.labelsize", 14)  # Set font size for yaxis tick labels
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

    subsetCountOrderedIntervals = 50
    subsetCountSharpness = None
    subsetCountCalibration = None
    subsetCountAdversarialCalibration = None
    subsetCountResidualUncertainty = 200
    subsetCountParity = 200

    binCount = 100
    
    fig, axs = plt.subplots(1, 1, figsize=(4.5, 4.5), sharex = False, sharey = False)
    
    print(axs)
    
    # Format x-axis to be percentage tick marks
    axs.xaxis.set_major_formatter(PercentFormatter(1.0))

    # Make calibration plot
    epistemicLabel = "Evidential Regression\n(Epistemic, " + lambdaSymbolStr + " = " + str(lamb) + ")"
    aleatoricLabel = "Evidential Regression\n(Aleatoric, " + lambdaSymbolStr + " = " + str(lamb) + ")"
    axs = uct.viz_Ensemble.plot_adversarial_group_calibration(pred_mean, pred_std, target, leg_loc=4, leg_labels=["5-fold Ensemble", "MC Dropout", epistemicLabel, "5-fold Ensemble", "MC Dropout", epistemicLabel], 
            colorList=["#1f77b4", "darkorange", "darkgreen", "#1f77b4", "darkorange", "darkgreen"], markerList=["o", "s", "p", "o", "s", "p"], lineList=["dashed", "dashed", "dashed", "solid", "solid", "solid"], cali_type="miscal_area", 
            seed=subsetSeed, showTitle=showTitleChoice, showLegend=showLegendChoice, ax=axs)
    axs.yaxis.set_ticks(np.arange(0.0, 0.6, 0.1))
    axs.xaxis.set_ticks(np.arange(0.0, 1.2, 0.2))
    axs.set_ylim(0.0, 0.51)
    axs.set_xlabel("Group Size (% of Test Set)")

    # Adjust subplots spacing
    #fig.subplots_adjust(wspace=0.75)
    #fig.subplots_adjust(hspace=0.50)
    plt.tight_layout()
    axs.grid(visible=None)
    axs.tick_params(axis="both", which="major", direction="out", color="black", length=4.0, width=2.0)

    savefig=True
    # Save figure
    if savefig:
        uct.viz_Evidential.save_figure(plot_save_str, "svg", white_background=True)
        uct.viz_Evidential.save_figure(plot_save_str, "png", white_background=True)

targetFilePath = "VALID_TARGETS_FILEPATH"

predictionFilePathList = []
predictionFilePathEnsemble = "ENSEMBLE_PREDICTIONS_FILEPATH"
predictionFilePathDropout = "MC_DROPOUT_PREDICTIONS_FILEPATH"
predictionFilePathEvidential = "EVIDENTIAL_PREDICTIONS_FILEPATH"
predictionFilePathList.append(predictionFilePathEnsemble)
predictionFilePathList.append(predictionFilePathDropout)
predictionFilePathList.append(predictionFilePathEvidential)

uncertaintyFilePathList = []
uncertaintyFilePathEnsemble = "ENSEMBLE_RECALIBRATED_UNCERTAINTY_FILEPATH"
uncertaintyFilePathDropout = "MC_DROPOUT_RECALIBRATED_UNCERTAINTY_FILEPATH"
uncertaintyFilePathEvidential = "EVIDENTIAL_RECALIBRATED_UNCERTAINTY_FILEPATH"
uncertaintyFilePathList.append(uncertaintyFilePathEnsemble)
uncertaintyFilePathList.append(uncertaintyFilePathDropout)
uncertaintyFilePathList.append(uncertaintyFilePathEvidential)

print(predictionFilePathList)

# Where to save plot
savePlotPath = "ADVERSIAL_GROUP_CALIBRATION_RECALIBRATION_PLOT_SAVE_FILEPATH_USER_SPECIFIED"

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
predictionFileNameList.append(predictionFileNameEnsemble)
predictionFileNameList.append(predictionFileNameDropout)
predictionFileNameList.append(predictionFileNameEvidential)

uncertaintyFileNameList = []
uncertaintyFileNameEnsemble = "IS2RE_all_CGCNN_ValID_Ensemble5_Energy_StdDev.csv"
uncertaintyFileNameDropout = "IS2RE_all_CGCNN_ValID_MC_Energy_StdDev1000.csv"
epistemicUncertaintyFileNameEvidential = "epistemic_uncertainty_" + jobID + ".csv"
aleatoricUncertaintyFileNameEvidential = "aleatoric_uncertainty_" + jobID + ".csv"
uncertaintyFileNameList.append(uncertaintyFileNameEnsemble)
uncertaintyFileNameList.append(uncertaintyFileNameDropout)
uncertaintyFileNameList.append(epistemicUncertaintyFileNameEvidential)

print(uncertaintyFileNameList)

recalibratedUncertaintyFileNameList = []
recalibratedUncertaintyFileNameEnsemble = "IS2RE_all_CGCNN_ValID_Ensemble5_Energy_StdDev_RecalScalar.csv"
recalibratedUncertaintyFileNameDropout = "IS2RE_all_CGCNN_ValID_MC_Energy_StdDev1000_RecalScalar.csv"
recalibratedEpistemicUncertaintyFileNameEvidential = "epistemic_Recalibrated_Scalar_" + jobID + ".csv"
recalibratedUncertaintyFileNameList.append(recalibratedUncertaintyFileNameEnsemble)
recalibratedUncertaintyFileNameList.append(recalibratedUncertaintyFileNameDropout)
recalibratedUncertaintyFileNameList.append(recalibratedEpistemicUncertaintyFileNameEvidential)


predictionFileList = []
uncertaintyFileList = []
recalibratedUncertaintyFileList = []
for ind in range(len(predictionFilePathList)):
    predictionFilePath = os.path.join(predictionFilePathList[ind], predictionFileNameList[ind])
    uncertaintyFilePath = os.path.join(uncertaintyFilePathList[ind], uncertaintyFileNameList[ind])
    recalibratedUncertaintyFilePath = os.path.join(uncertaintyFilePathList[ind] + "recalibration/", recalibratedUncertaintyFileNameList[ind])

    predictionFileList.append(predictionFilePath)
    uncertaintyFileList.append(uncertaintyFilePath)
    recalibratedUncertaintyFileList.append(recalibratedUncertaintyFilePath)


print("\nPrediction Files:")
print(predictionFileList)
print("\nUncertainty Files:")
print(uncertaintyFileList)
print("\nRecalibrated Uncertainty Files:")
print(recalibratedUncertaintyFileList)

targetFile = targetFilePath + targetFileName
# What to name plot
savePlotNameTitle = "UQ_Technique_Compare_Overlay_AdvCalibration_Recal_Scalar_Title_Seed" + str(rngVisualSeed)
savePlotNameLeg = "UQ_Technique_Compare_Overlay_AdvCalibration_Recal_Scalar_Leg_Seed" + str(rngVisualSeed)
savePlotNameLegTitle = "UQ_Technique_Compare_Overlay_AdvCalibration_Recal_Scalar_LegTitle_Seed" + str(rngVisualSeed)
savePlotName = "UQ_Technique_Compare_Overlay_AdvCalibration_Recal_Scalar_Seed" + str(rngVisualSeed)

savePlotFileTitle = os.path.join(savePlotPath, savePlotNameTitle)
savePlotFileLeg = os.path.join(savePlotPath, savePlotNameLeg)
savePlotFileLegTitle = os.path.join(savePlotPath, savePlotNameLegTitle)
savePlotFile = os.path.join(savePlotPath, savePlotName)

targetPD = pd.read_csv(targetFile)


predictionPDList = []
uncertaintyPDList = []
recalibratedUncertaintyPDList = []

for ind in range(len(predictionFileList)):
    predictionPD = pd.read_csv(predictionFileList[ind])
    predictionPDList.append(predictionPD)
    uncertaintyPD = pd.read_csv(uncertaintyFileList[ind])
    uncertaintyPDList.append(uncertaintyPD)
    recalibratedUncertaintyPD = pd.read_csv(recalibratedUncertaintyFileList[ind])
    recalibratedUncertaintyPDList.append(recalibratedUncertaintyPD)

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

predictionNPEnsemble = predictionPDList[0][predictionColNameEnsemble].to_numpy()
predictionNPDropout = predictionPDList[1][predictionColNameDropout].to_numpy()
predictionNPEvidential = predictionPDList[2][predictionColNameEvidential].to_numpy()
predictionNPList.append(predictionNPEnsemble)
predictionNPList.append(predictionNPDropout)
predictionNPList.append(predictionNPEvidential)
predictionNPList.append(predictionNPEnsemble)
predictionNPList.append(predictionNPDropout)
predictionNPList.append(predictionNPEvidential)

uncertaintyNPEnsemble = uncertaintyPDList[0][uncertaintyColNameEnsemble].to_numpy()
uncertaintyNPDropout = uncertaintyPDList[1][uncertaintyColNameDropout].to_numpy()
epistemicUncertaintyNPEvidential = uncertaintyPDList[2][epistemicUncertaintyColNameEvidential].to_numpy()

uncertaintyNPList.append(uncertaintyNPEnsemble)
uncertaintyNPList.append(uncertaintyNPDropout)
uncertaintyNPList.append(epistemicUncertaintyNPEvidential)

recalibratedUncertaintyNPEnsemble = recalibratedUncertaintyPDList[0][uncertaintyColNameEnsemble].to_numpy()
recalibratedUncertaintyNPDropout = recalibratedUncertaintyPDList[1][uncertaintyColNameDropout].to_numpy()
recalibratedEpistemicUncertaintyNPEvidential = recalibratedUncertaintyPDList[2][epistemicUncertaintyColNameEvidential].to_numpy()
uncertaintyNPList.append(recalibratedUncertaintyNPEnsemble)
uncertaintyNPList.append(recalibratedUncertaintyNPDropout)
uncertaintyNPList.append(recalibratedEpistemicUncertaintyNPEvidential)

# Whether the title and/or legend is going to be shown.
make_plots(predictionNPList, uncertaintyNPList, targetNP, lamb, rngVisualSeed, lineplot_array = None, lineplot_legLabels = None, lineplot_xLabels = None,
        showTitleChoice=True, showLegendChoice=False, plot_save_str = savePlotFileTitle)
make_plots(predictionNPList, uncertaintyNPList, targetNP, lamb, rngVisualSeed, lineplot_array = None, lineplot_legLabels = None, lineplot_xLabels = None,
        showTitleChoice=False, showLegendChoice=True, plot_save_str = savePlotFileLeg)
make_plots(predictionNPList, uncertaintyNPList, targetNP, lamb, rngVisualSeed, lineplot_array = None, lineplot_legLabels = None, lineplot_xLabels = None,
        showTitleChoice=True, showLegendChoice=True, plot_save_str = savePlotFileLegTitle)
make_plots(predictionNPList, uncertaintyNPList, targetNP, lamb, rngVisualSeed, lineplot_array = None, lineplot_legLabels = None, lineplot_xLabels = None,
        showTitleChoice=False, showLegendChoice=False, plot_save_str = savePlotFile)
