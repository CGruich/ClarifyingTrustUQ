import uncertainty_toolbox as uct
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Save the unicode string for the lambda symbol so we can put it in the plots later.
# Make this a global variable such that it is accessible from all functions.
global lambdaSymbolStr
lambdaSymbolStr = "\u03bb"

def make_plots(pred_mean, pred_std, target, lamb, subsetSeed, lineplot_array = None, lineplot_legLabels = None, lineplot_xLabels = None, showTitleChoice = True, showLegendChoice = True, plot_save_str="row"):
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

    #ylims = [-20, 20]
    
    subsetCountOrderedIntervals = 50
    subsetCountSharpness = None
    subsetCountCalibration = None
    subsetCountAdversarialCalibration = None
    subsetCountResidualUncertainty = 200
    subsetCountParity = 200

    binCount = 100
    
    fig, axs = plt.subplots(4, 3, figsize=(24, 20))

    # Figure supertitle
    fig.suptitle("\nUncertainty Quantification Plots\n(Evidential Regression, " + lambdaSymbolStr + " = " + str(lamb) + ")", fontsize=24)
    
    print(axs)

    # Make ordered intervals plot
    axs[0][0] = uct.viz_Evidential.plot_intervals_ordered(
        pred_mean, pred_std, target, lamb, n_subset=subsetCountOrderedIntervals, seed=subsetSeed, showTitle=showTitleChoice, showLegend=showLegendChoice, ax=axs[0][0]
    )

    # Make calibration plot
    axs[0][1] = uct.viz_Evidential.plot_calibration(pred_mean, pred_std, target, lamb, seed=subsetSeed, showTitle=showTitleChoice, ax=axs[0][1])

    # Make sharpness plot
    axs[0][2] = uct.viz_Evidential.plot_sharpness(pred_mean, pred_std, target, lamb, subsetCountSharpness, bins=binCount, seed=subsetSeed, showTitle=showTitleChoice, ax=axs[0][2])
    
    # Make residuals vs. uncertainties plot
    axs[1][0] = uct.viz_Evidential.plot_residuals_vs_stds(pred_mean, pred_std, target, lamb, subsetCountResidualUncertainty, seed=subsetSeed, showTitle=showTitleChoice, showLegend=showLegendChoice, ax=axs[1][0])
    
    # Plot adversarial group calibration curve.
    axs[3][0] = uct.viz_Evidential.plot_adversarial_group_calibration(pred_mean, pred_std, target, lamb, subsetCountAdversarialCalibration, seed=subsetSeed, showTitle=showTitleChoice, ax=axs[3][0])

    # Make a parity plot, this shows model accuracy on a sample-by-sample basis
    axs[2][0] = uct.viz_Evidential.plot_parity(pred_mean, pred_std, target, lamb, subsetCountParity, leg_loc=4, seed=subsetSeed, showLegend=showLegendChoice, ax=axs[2][0])
    
    gridSizeCount = 100
    # Make a parity plot, but with hexagonal binning. This shows model accuracy on a sample-by-sample basis.
    axs[2][1] = uct.viz_Evidential.plot_parity_hexagonal(pred_mean, pred_std, target, lamb, gridSize=gridSizeCount, subFig=fig, n_subset = None, leg_loc=4, seed=subsetSeed, showTitle=showTitleChoice, showLegend=showLegendChoice, ax=axs[2][1])

    # Hexagonal plot, but with log color scale and bins
    axs[2][2] = uct.viz_Evidential.plot_parity_hexagonal(pred_mean, pred_std, target, lamb, gridSize=gridSizeCount, bins="log", subFig=fig, n_subset = None, leg_loc=4, seed=subsetSeed, showTitle=showTitleChoice, showLegend=showLegendChoice, ax=axs[2][2])

    axs[1][1] = uct.viz_Evidential.plot_residuals_vs_stds_hexagonal(pred_mean, pred_std, target, lamb, gridSize=gridSizeCount, bins="log", subFig=fig, n_subset = None, leg_loc=4, seed=subsetSeed, showTitle=showTitleChoice, showLegend=showLegendChoice, ax=axs[1][1])

    # Violin + Box plot
    axs[1][2] = uct.viz_Evidential.plot_violin_box_plot(pred_std, lamb, ylims=None, leg_loc=4, seed=subsetSeed, kdePoints=1000, width=0.5, showTitle=showTitleChoice, showLegend=showLegendChoice, showMeans=False, showExtrema=False, showMedians=False, showVertical=True, ax=axs[1][2])

    if type(lineplot_array) != type(None):
        # Accuracy lineplot to assess MC dropout accuracy at different dropout rates
        axs[3][1] = uct.viz_Evidential.plot_accuracy_lineplot(lineplot_array, leg_loc = 2, leg_labels = lineplot_legLabels, xLabels = lineplot_xLabels, showTitle=showTitleChoice, showLegend=showLegendChoice, ax=axs[3][1])

    # Adjust subplots spacing
    fig.subplots_adjust(wspace=0.25)
    fig.subplots_adjust(hspace=0.50)

    # Save figure
    if savefig:
        uct.viz_Evidential.save_figure(plot_save_str, "svg", white_background=True)
        uct.viz_Evidential.save_figure(plot_save_str, "png", white_background=True)

targetFilePath = "VALID_TEST_TARGETS"
predictionFilePath = "EVIDENTIAL_PREDICTIONS"
uncertaintyFilePath = "EVIDENTIAL_UNCERTAINTY"
# Where to save plot
savePlotPath = uncertaintyFilePath

xLabelList = ["0", "0.05", "0.1", "0.15", "0.2"]
lineplotDataArray = np.asarray([(0.0, 0.633265837888014), (0.05, 0.629793262621284), (0.1, 0.631996560722256), (0.15, 0.632487070559222), (0.2, 0.633375750628576)], dtype = "float32")

# Random number generation seed for visualization
rngVisualSeed = 100

# Lambda hyperparameter used for DER
lamb = 0.1
jobID = "9189842"
lambStr = str(lamb).replace(".", "p")

targetFileName = "targets.csv"
predictionFileName = "evidential_gamma_" + jobID + ".csv"
aleatoricUncertaintyFileName = "aleatoric_uncertainty_" + jobID + ".csv"
epistemicUncertaintyFileName = "epistemic_uncertainty_" + jobID + ".csv"

# What to name plot
aleatoricSavePlotName = "Aleatoric_Lamb" + lambStr + "_Seed" + str(rngVisualSeed)
epistemicSavePlotName = "Epistemic_Lamb" + lambStr + "_Seed" + str(rngVisualSeed)

targetFile = targetFilePath + targetFileName
predictionFile = predictionFilePath + predictionFileName
aleatoricUncertaintyFile = uncertaintyFilePath + aleatoricUncertaintyFileName
epistemicUncertaintyFile = uncertaintyFilePath + epistemicUncertaintyFileName

aleatoricSavePlot = savePlotPath + aleatoricSavePlotName
epistemicSavePlot = savePlotPath + epistemicSavePlotName

targetPD = pd.read_csv(targetFile)
predictionPD = pd.read_csv(predictionFile)
aleatoricUncertaintyPD = pd.read_csv(aleatoricUncertaintyFile)
epistemicUncertaintyPD = pd.read_csv(epistemicUncertaintyFile)

targetColName = "Target (eV)"
predictionColName = "Ads. Energy. (eV)"
aleatoricUncertaintyColName = "Aleatoric Uncertainty (std. dev., eV)"
epistemicUncertaintyColName = "Epistemic Uncertainty (std. dev., eV)"

print("\nTarget Dataframe: ")
print(targetPD)
print("\nPredicton Dataframe: ")
print(predictionPD)
print("\nPrediction Aleatoric Uncertainty Dataframe: ")
print(aleatoricUncertaintyPD)
print("\nPrediction Epistemic Uncertainty Dataframe: ")
print(epistemicUncertaintyPD)

print("\nConverting to NumPy arrays...")

targetNP = targetPD[targetColName].to_numpy()
predictionNP = predictionPD[predictionColName].to_numpy()
aleatoricUncertaintyNP = aleatoricUncertaintyPD[aleatoricUncertaintyColName].to_numpy()
epistemicUncertaintyNP = epistemicUncertaintyPD[epistemicUncertaintyColName].to_numpy()

print("\nTarget Array: ")
print(targetNP)
print("\nPrediction Array: ")
print(predictionNP)
print("\nPrediction Aleatoric Uncertainty Array: ")
print(aleatoricUncertaintyNP)
print("\nPrediction Epistemic Uncertainty Array: ")
print(epistemicUncertaintyNP)

# Save figure specification
savefig = True

aleatoricMace = uct.mean_absolute_calibration_error(predictionNP, aleatoricUncertaintyNP, targetNP)
aleatoricRmsce = uct.root_mean_squared_calibration_error(predictionNP, aleatoricUncertaintyNP, targetNP)
aleatoricMa = uct.miscalibration_area(predictionNP, aleatoricUncertaintyNP, targetNP)

epistemicMace = uct.mean_absolute_calibration_error(predictionNP, epistemicUncertaintyNP, targetNP)
epistemicRmsce = uct.root_mean_squared_calibration_error(predictionNP, epistemicUncertaintyNP, targetNP)
epistemicMa = uct.miscalibration_area(predictionNP, epistemicUncertaintyNP, targetNP)

# Whether the title and/or legend is going to be shown.
showTitleChoice = True
showLegendChoice = True

make_plots(predictionNP, aleatoricUncertaintyNP, targetNP, lamb, rngVisualSeed, lineplotDataArray, lineplot_legLabels = None, lineplot_xLabels = xLabelList, 
        showTitleChoice=showTitleChoice, showLegendChoice=showLegendChoice, plot_save_str = aleatoricSavePlot)
make_plots(predictionNP, epistemicUncertaintyNP, targetNP, lamb, rngVisualSeed, lineplotDataArray, lineplot_legLabels = None, lineplot_xLabels = xLabelList, 
        showTitleChoice=showTitleChoice, showLegendChoice=showLegendChoice, plot_save_str = epistemicSavePlot)

aleatoricMetrics = uct.metrics.get_all_metrics(predictionNP, aleatoricUncertaintyNP, targetNP)
epistemicMetrics = uct.metrics.get_all_metrics(predictionNP, epistemicUncertaintyNP, targetNP)

