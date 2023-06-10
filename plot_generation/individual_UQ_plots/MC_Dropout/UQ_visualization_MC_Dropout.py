import uncertainty_toolbox as uct
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def make_plots(pred_mean, pred_std, target, subsetSeed, lineplot_array = None, lineplot_legLabels = None, lineplot_xLabels = None, showTitleChoice = True, showLegendChoice = True, plot_save_str="row"):
    """Make set of plots."""

    # "subsetSeed" is for plotting reproducibility any time data is binned (i.e. collected and averaged) for the purposes of visualization.
    # Turn on plot style customization
    uct.viz_MC_Dropout.set_style()
    # Change font size in general
    uct.viz_MC_Dropout.update_rc("font.size", 14)  # Set font size
    # Change font fize of the title
    #uct.viz_MC_Dropout.update_rc("figure.titlesize", 16) # Set title font size
    # Change xtick label size
    uct.viz_MC_Dropout.update_rc("xtick.labelsize", 14)  # Set font size for xaxis tick labels
    # Change ytick label size
    uct.viz_MC_Dropout.update_rc("ytick.labelsize", 14)  # Set font size for yaxis tick labels
    # Turn off Latex mode
    uct.viz_MC_Dropout.update_rc("text.usetex", False)
    # Update legend general font size
    uct.viz_MC_Dropout.update_rc("legend.fontsize", 10)
    # Configure legend title font size
    uct.viz_MC_Dropout.update_rc("legend.title_fontsize", "small")
    # Poster quality plots
    uct.viz_MC_Dropout.update_rc("figure.dpi", 600)
    # Change the font globally
    uct.viz_MC_Dropout.update_rc("font.family", "Times New Roman")
    # Change the figure border width globally
    uct.viz_MC_Dropout.update_rc("axes.linewidth", 2.0)
    # Change the figure border color globally
    uct.viz_MC_Dropout.update_rc("axes.edgecolor", "black")

    subsetCountOrderedIntervals = 50
    subsetCountSharpness = None
    subsetCountCalibration = None
    subsetCountAdversarialCalibration = None
    subsetCountResidualUncertainty = 200
    subsetCountParity = 200

    binCount = 100

    fig, axs = plt.subplots(4, 3, figsize=(20, 20))
    
    # Figure supertitle
    fig.suptitle("\nUncertainty Quantification Plots\n(MC Dropout)", fontsize=24)
    
    print(axs)

    # Make ordered intervals plot
    axs[0][0] = uct.viz_MC_Dropout.plot_intervals_ordered(
        pred_mean, pred_std, target, n_subset=subsetCountOrderedIntervals, seed=subsetSeed, showTitle=showTitleChoice, showLegend=showLegendChoice, ax=axs[0][0]
    )

    # Make calibration plot
    axs[0][1] = uct.viz_MC_Dropout.plot_calibration(pred_mean, pred_std, target, seed=subsetSeed, showTitle=showTitleChoice, ax=axs[0][1])

    # Make sharpness plot
    axs[0][2] = uct.viz_MC_Dropout.plot_sharpness(pred_mean, pred_std, target, subsetCountSharpness, bins=binCount, seed=subsetSeed, showTitle=showTitleChoice, ax=axs[0][2])
    
    # Make residuals vs. uncertainties plot
    axs[1][0] = uct.viz_MC_Dropout.plot_residuals_vs_stds(pred_mean, pred_std, target, subsetCountResidualUncertainty, seed=subsetSeed, showTitle=showTitleChoice, showLegend=showLegendChoice, ax=axs[1][0])
    
    # Plot adversarial group calibration curve.
    axs[3][0] = uct.viz_MC_Dropout.plot_adversarial_group_calibration(pred_mean, pred_std, target, subsetCountAdversarialCalibration, seed=subsetSeed, showTitle=showTitleChoice, ax=axs[3][0])

    # Make a parity plot, this shows model accuracy on a sample-by-sample basis
    axs[2][0] = uct.viz_MC_Dropout.plot_parity(pred_mean, pred_std, target, subsetCountParity, leg_loc=4, seed=subsetSeed, showTitle=showTitleChoice, showLegend=showLegendChoice, ax=axs[2][0])

    gridSizeCount = 100
    # Make a parity plot, but with hexagonal binning. This shows model accuracy on a sample-by-sample basis.
    axs[2][1] = uct.viz_MC_Dropout.plot_parity_hexagonal(pred_mean, pred_std, target, gridSize=gridSizeCount, subFig=fig, n_subset = None, leg_loc=4, seed=subsetSeed, showTitle=showTitleChoice, showLegend=showLegendChoice, ax=axs[2][1])

    # Hexagonal accuracy plot, but with log color scale and bins
    axs[2][2] = uct.viz_MC_Dropout.plot_parity_hexagonal(pred_mean, pred_std, target, gridSize=gridSizeCount, bins="log", subFig=fig, n_subset = None, leg_loc=4, seed=subsetSeed, showTitle=showTitleChoice, showLegend=showLegendChoice, ax=axs[2][2])

    # Residual Error vs Uncertainty but with hexagonal binning
    axs[1][1] = uct.viz_MC_Dropout.plot_residuals_vs_stds_hexagonal(pred_mean, pred_std, target, gridSize=gridSizeCount, bins="log", subFig=fig, n_subset = None, leg_loc=4, seed=subsetSeed, showTitle=showTitleChoice, showLegend=showLegendChoice, ax=axs[1][1])
    
    # Violin/Box plot
    axs[1][2] = uct.viz_MC_Dropout.plot_violin_box_plot(pred_std, ylims=None, leg_loc=4, seed=subsetSeed, kdePoints=1000, width=0.5, showTitle=showTitleChoice, showLegend=showLegendChoice, showMeans=False, showExtrema=False, showMedians=False, showVertical=True, ax=axs[1][2])

    if type(lineplot_array) != type(None):
        # Accuracy lineplot to assess MC dropout accuracy at different dropout rates
        axs[3][1] = uct.viz_MC_Dropout.plot_accuracy_lineplot(lineplot_array, leg_loc = 2, leg_labels = lineplot_legLabels, xLabels = lineplot_xLabels, showTitle=showTitleChoice, showLegend=showLegendChoice, ax=axs[3][1])

    # Adjust subplots spacing
    fig.subplots_adjust(wspace=0.25)
    fig.subplots_adjust(hspace=0.50)

    # Save figure
    if savefig:
        uct.viz_MC_Dropout.save_figure(plot_save_str, "svg", white_background=True)
        uct.viz_MC_Dropout.save_figure(plot_save_str, "png", white_background=True)

targetFilePath = "VALID_TEST_TARGETS"
predictionFilePath = "MC_DROPOUT_PREDICTIONS"
uncertaintyFilePath = "MC_DROPOUT_UNCERTAINTY"

# Where to save plot
savePlotPath = uncertaintyFilePath

xLabelList = ["0%", "5%", "10%", "20%", "30%"]
lineplotDataArray = np.asarray([(0, 0.650565663721069), (5, 0.656844056111952), (10, 0.6692073030924), (20, 0.742656429820688), (30, 0.890508384605024)], dtype = "float32")

# Random number generation seed for visualization
rngVisualSeed = 100

targetFileName = "targets.csv"
predictionFileName = "IS2RE_all_CGCNN_ValID_MC_Energy_Mean1000.csv"
uncertaintyFileName = "IS2RE_all_CGCNN_ValID_MC_Energy_StdDev1000.csv"
# What to name plot
savePlotName = "MC_Dropout_1000_Row_Seed" + str(rngVisualSeed)

targetFile = targetFilePath + targetFileName
predictionFile = predictionFilePath + predictionFileName
uncertaintyFile = uncertaintyFilePath + uncertaintyFileName
savePlot = savePlotPath + savePlotName

targetPD = pd.read_csv(targetFile)
predictionPD = pd.read_csv(predictionFile)
uncertaintyPD = pd.read_csv(uncertaintyFile)

targetColName = "Target (eV)"
predictionColName = "Mean Sample Energy (eV)"
uncertaintyColName = "Std. Dev. Sample Energy (eV)"

print("\nTarget Dataframe: ")
print(targetPD)
print("\nPredicton Dataframe: ")
print(predictionPD)
print("\nPrediction Uncertainty Dataframe: ")
print(uncertaintyPD)

print("\nConverting to NumPy arrays...")

targetNP = targetPD[targetColName].to_numpy()
predictionNP = predictionPD[predictionColName].to_numpy()
uncertaintyNP = uncertaintyPD[uncertaintyColName].to_numpy()

print("\nTarget Array: ")
print(targetNP)
print("\nPrediction Array: ")
print(predictionNP)
print("\nPrediction Uncertainty Array: ")
print(uncertaintyNP)

# Save figure specification
savefig = True

mace = uct.mean_absolute_calibration_error(predictionNP, uncertaintyNP, targetNP)
rmsce = uct.root_mean_squared_calibration_error(predictionNP, uncertaintyNP, targetNP)
ma = uct.miscalibration_area(predictionNP, uncertaintyNP, targetNP)

# Whether to show the title and/or legend
showTitleChoice = True
showLegendChoice = True
make_plots(predictionNP, uncertaintyNP, targetNP, rngVisualSeed, lineplotDataArray, lineplot_legLabels = None, lineplot_xLabels = xLabelList, 
        showTitleChoice=showTitleChoice, showLegendChoice=showLegendChoice, plot_save_str = savePlot)
metrics = uct.metrics.get_all_metrics(predictionNP, uncertaintyNP, targetNP)

