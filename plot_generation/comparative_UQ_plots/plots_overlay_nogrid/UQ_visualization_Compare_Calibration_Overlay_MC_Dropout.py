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

def make_plots(pred_mean, pred_std, target, lamb, subsetSeed, sampleList = None, lineplot_array = None, lineplot_legLabels = None, lineplot_xLabels = None, showTitleChoice = True, showLegendChoice = True, plot_save_str="row"):
    """Make set of plots."""

    # "subsetSeed" is for plotting reproducibility any time data is binned (i.e. collected and averaged) for the purposes of visualization.
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

    #ylims = [-20, 20]
    
    subsetCountOrderedIntervals = 50
    subsetCountSharpness = None
    subsetCountCalibration = None
    subsetCountAdversarialCalibration = None
    subsetCountResidualUncertainty = 200
    subsetCountParity = 200

    binCount = 100
    
    fig, axs = plt.subplots(1, 1, figsize=(3.8, 3.8), sharex = False, sharey = False)
    
    print(axs)

    # Make calibration plot
    axs = uct.viz_MC_Dropout.plot_calibration(pred_mean, pred_std, target, leg_loc=2, leg_labels=sampleList, legendTitle = "(Samples, Miscalibration Area)", seed=subsetSeed, showTitle=showTitleChoice, showLegend=showLegendChoice, 
            alphaColorMap=True, showAlpha=True, ax=axs)
    axs.yaxis.set_ticks(np.arange(0.0, 1.2, 0.2))
    axs.xaxis.set_ticks(np.arange(0.0, 1.2, 0.2))
    if showTitleChoice == True:
        axs.set(title = "MC Dropout\nCalibration Equilibration")
    
    # Adjust subplots spacing
    #fig.subplots_adjust(wspace=0.75)
    #fig.subplots_adjust(hspace=0.50)
    plt.tight_layout()
    axs.grid(visible=None)
    axs.tick_params(axis="both", which="major", direction="out", color="black", length=4.0, width=2.0)
    #minorTicksLocator = AutoMinorLocator(2)
    #axs.xaxis.set_minor_locator(minorTicksLocator)
    #axs.yaxis.set_minor_locator(minorTicksLocator)
    #axs.tick_params(axis="both", which="minor", direction="out", color="black", length=3.0, width=1.25)

    # Save figure
    if savefig:
        uct.viz_Evidential.save_figure(plot_save_str, "svg", white_background=True)
        uct.viz_Evidential.save_figure(plot_save_str, "png", white_background=True)

targetFilePath = "VALID_TARGETS_FILEPATH"

predictionFilePathList = []
predictionFilePathDropout = "MC_DROPOUT_PREDICTIONS_FILEPATH"
predictionFilePathList.append(predictionFilePathDropout)

uncertaintyFilePathList = []
uncertaintyFilePathDropout = "MC_DROPOUT_UNCERTAINTY_FILEPATH"
uncertaintyFilePathList.append(uncertaintyFilePathDropout)

print(predictionFilePathList)

# Where to save plot
savePlotPath = "MC_DROPOUT_OVERLAY_PLOT_SAVE_FILEPATH"

# Create directory to save the plot to, if it does not exist
if not os.path.exists(savePlotPath):
    os.makedirs(savePlotPath)

# Random number generation seed for visualization
rngVisualSeed = 100

# Lambda hyperparameter used for DER
lamb = 0.1

targetFileName = "targets.csv"

predictionFileNameList = []
predictionFileNameDropout = "IS2RE_all_CGCNN_ValID_MC_Energy_Mean"
predictionFileNameList.append(predictionFileNameDropout)


uncertaintyFileNameList = []
uncertaintyFileNameDropout = "IS2RE_all_CGCNN_ValID_MC_Energy_StdDev"
uncertaintyFileNameList.append(uncertaintyFileNameDropout)

print(uncertaintyFileNameList)

predictionFileList = []
uncertaintyFileList = []

# Different MC dropout sample sizes to plot together,
sampleList = ["5", "10", "20", "30", "50", "1000"]

for ind in range(len(sampleList)):
    predictionFilePath = os.path.join(predictionFilePathList[0], predictionFileNameList[0]) + sampleList[ind] + ".csv"
    uncertaintyFilePath = os.path.join(uncertaintyFilePathList[0], uncertaintyFileNameList[0]) + sampleList[ind] + ".csv"

    predictionFileList.append(predictionFilePath)
    uncertaintyFileList.append(uncertaintyFilePath)


print("\nPrediction Files:")
print(predictionFileList)
print("\nUncertainty Files:")
print(uncertaintyFileList)


targetFile = targetFilePath + targetFileName
# What to name plot
savePlotNameTitle = "UQ_Technique_Compare_Overlay_Calibration_MC_Dropout_Title_Seed" + str(rngVisualSeed)
savePlotNameLeg = "UQ_Technique_Compare_Overlay_Calibration_MC_Dropout_Leg_Seed" + str(rngVisualSeed)
savePlotNameLegTitle = "UQ_Technique_Compare_Overlay_Calibration_MC_Dropout_LegTitle_Seed" + str(rngVisualSeed)
savePlotName = "UQ_Technique_Compare_Overlay_Calibration_MC_Dropout_Seed" + str(rngVisualSeed)

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
uncertaintyColNameEnsemble = "Std. Dev. Sample Energy (eV)"
uncertaintyColNameDropout = uncertaintyColNameEnsemble

targetNP = targetPD[targetColName].to_numpy()

predictionNPList = []
uncertaintyNPList = []

for ind in range(len(sampleList)):
    predictionNPDropout = predictionPDList[ind][predictionColNameDropout].to_numpy()
    predictionNPList.append(predictionNPDropout)

    uncertaintyNPDropout = uncertaintyPDList[ind][uncertaintyColNameDropout].to_numpy()
    uncertaintyNPList.append(uncertaintyNPDropout)

# Save figure specification
savefig = True

print(predictionNPList)
print(uncertaintyNPList)
# Whether the title and/or legend is going to be shown.
make_plots(predictionNPList, uncertaintyNPList, targetNP, lamb, rngVisualSeed, sampleList=sampleList, lineplot_array = None, lineplot_legLabels = None, lineplot_xLabels = None,
        showTitleChoice=True, showLegendChoice=False, plot_save_str = savePlotFileTitle)
make_plots(predictionNPList, uncertaintyNPList, targetNP, lamb, rngVisualSeed, sampleList=sampleList, lineplot_array = None, lineplot_legLabels = None, lineplot_xLabels = None,
        showTitleChoice=False, showLegendChoice=True, plot_save_str = savePlotFileLeg)
make_plots(predictionNPList, uncertaintyNPList, targetNP, lamb, rngVisualSeed, sampleList=sampleList, lineplot_array = None, lineplot_legLabels = None, lineplot_xLabels = None,
        showTitleChoice=True, showLegendChoice=True, plot_save_str = savePlotFileLegTitle)
make_plots(predictionNPList, uncertaintyNPList, targetNP, lamb, rngVisualSeed, sampleList=sampleList, lineplot_array = None, lineplot_legLabels = None, lineplot_xLabels = None,
        showTitleChoice=False, showLegendChoice=False, plot_save_str = savePlotFile)
