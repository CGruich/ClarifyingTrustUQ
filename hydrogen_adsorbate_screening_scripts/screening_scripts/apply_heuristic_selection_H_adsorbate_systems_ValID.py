import os.path
import lmdb
import numpy as np
import pandas as pd
import torch
from ocpmodels.datasets import LmdbDataset


def apply_heuristic_subselection(
    y_pred_and_std,
    sidList,
    y_pred_Lower,
    y_pred_Upper,
    stdDevTarget,
    y_pred_Label,
    y_std_Label,
    sid_Label='sid',
):
    """Applies UQ-guided heuristics to select adsorbate/alloy systems for further study.

    Arguments:
        y_pred_and_std: PANDAS Dataframe with prediction and associated uncertainty estimate for a UQ technique.
        sidList: A list of system IDs that identify which systems to apply the heuristic/selection to.
        y_pred_Lower: Select predictions lower bound
        y_pred_Upper: Select predictions upper bound
        stdDevTarget: Select predictions with an associated uncertainty (std. dev.) less than or equal to target
        y_pred_Label: Column label for predictions
        y_std_Label: Column label for uncertainty estimates of predictions
        sid_Label: Column label for sid or "system ID", the unique integer identifier used for adsorbate/alloy systems
    Returns:
        relevantSystems: PANDAS Dataframe of sid/prediction/uncertainty of relevant systems matching heuristic criteria selection.
    """

    assert y_pred_Lower <= y_pred_Upper

    # Sub-select for systems we are interested in searching
    relevantSystems = y_pred_and_std.query(sid_Label + ' in @sidList')
    # Of the systems interested, select predictions within a lower bound and upper bound
    relevantSystems = relevantSystems[
        relevantSystems[y_pred_Label].between(y_pred_Lower, y_pred_Upper)
    ]
    print('Number of Predictions Matching Prediction Range: ' + str(relevantSystems.shape[0]))
    # Of the systems selected, further select systems with uncertainty less than or equal to a specific value
    relevantSystems = relevantSystems[relevantSystems[y_std_Label].between(0, stdDevTarget)]
    print(
        'Number of Predictions within Prediction Range <= Specified Uncertainty: '
        + str(relevantSystems.shape[0])
    )

    return relevantSystems


def retrieve_system_info(infoDF, sidList, sid_Label='sid'):
    """ Retrieves any placeholder system info for systems based on system ID.

    Arguments:
        infoDF: PANDAS Dataframe with relevant system info. Needs to have system id (sid) column
        sidList: Relevant system ids to pull
        sid_Label: Name of the system id (sid) column in infoDF.

    Returns:
        infoSubselectedDF: System info DF for the relevant sids
    """
    print(infoDF)
    print(sidList)
    infoSubselectDF = infoDF.query(sid_Label + ' in @sidList')
    print(infoSubselectDF)

    return infoSubselectDF


# Load the test target adsorption energies for calculation of the absolute errors via scikit-learn mean absolute error function
yTestPath = 'ValID_H_Adsorbate_data.lmdb'
# SinglePointLmdbDataset accepts .lmdb files, unlike TrajectoryLmdbDataset which accepts a whole directory of files
targetEnergiesLMDBS = LmdbDataset({'src': yTestPath})

# Get the energies and system ids of the test systems
# NumPy array
targetEnergiesNP = torch.tensor([data.y_relaxed for data in targetEnergiesLMDBS]).numpy()
# NumPy array
targetEnergiesSIDNP = torch.tensor([data.sid for data in targetEnergiesLMDBS]).numpy()
# List
targetEnergiesSID = list(targetEnergiesSIDNP)
# PANDAS Dataframes
targetEnergiesPD = pd.DataFrame(targetEnergiesNP)
targetEnergiesSIDPD = pd.DataFrame(targetEnergiesSIDNP)
targetEnergiesPD = pd.concat(
    [targetEnergiesSIDPD, targetEnergiesPD], axis=1, ignore_index=True
)
targetEnergiesPD.columns = ['sid', 'Target (eV)']
targetEnergiesPD['sid'] = targetEnergiesPD['sid'].astype(str)

predictionFilePathList = []
uncertaintyFilePathList = []
predictionFilePathEnsemble = 'ENSEMBLE_PREDICTION_FILEPATH'
predictionFilePathDropout = 'MC_DROPOUT_PREDICTION_FILEPATH'
predictionFilePathEvidential = 'EVIDENTIAL_PREDICTION_FILEPATH'
uncertaintyFilePathEnsemble = 'ENSEMBLE_RECALIBRATED_UNCERTAINTY_FILEPATH'
uncertaintyFilePathDropout = 'MC_DROPOUT_RECALIBRATED_UNCERTAINTY_FILEPATH'
epistemicUncertaintyFilePathEvidential = (
    'EVIDENTIAL_REGRESSION_RECALIBRATED_EPISTEMIC_UNCERTAINTY_FILEPATH'
)
predictionFilePathList.append(predictionFilePathEnsemble)
predictionFilePathList.append(predictionFilePathDropout)
predictionFilePathList.append(predictionFilePathEvidential)
uncertaintyFilePathList.append(uncertaintyFilePathEnsemble)
uncertaintyFilePathList.append(uncertaintyFilePathDropout)
uncertaintyFilePathList.append(epistemicUncertaintyFilePathEvidential)


# Random number generation seed for visualization
rngVisualSeed = 100

# Lambda hyperparameter used for DER
lamb = 0.05
jobID = '9600051'
lambStr = str(lamb).replace('.', 'p')

predictionFileNameList = []
uncertaintyFileNameList = []
predictionFileNameEnsemble = 'IS2RE_all_CGCNN_ValID_Ensemble5_Energy_Mean.csv'
predictionFileNameDropout = 'IS2RE_all_CGCNN_ValID_MC_Energy_Mean1000.csv'
predictionFileNameEvidential = 'evidential_gamma_' + jobID + '.csv'
uncertaintyFileNameEnsemble = 'IS2RE_all_CGCNN_ValID_Ensemble5_Energy_StdDev_RecalScalar.csv'
uncertaintyFileNameDropout = 'IS2RE_all_CGCNN_ValID_MC_Energy_StdDev1000_RecalScalar.csv'
epistemicUncertaintyFileNameEvidential = 'epistemic_Recalibrated_Scalar_' + jobID + '.csv'
predictionFileNameList.append(predictionFileNameEnsemble)
predictionFileNameList.append(predictionFileNameDropout)
predictionFileNameList.append(predictionFileNameEvidential)
uncertaintyFileNameList.append(uncertaintyFileNameEnsemble)
uncertaintyFileNameList.append(uncertaintyFileNameDropout)
uncertaintyFileNameList.append(epistemicUncertaintyFileNameEvidential)

systemLogPath = 'H_ADSORBATE_SYSTEMS_FROM_METADATA_FILEPATH'
systemLogName = 'oc20_H_adsorbate_systems.csv'
sidLogName = 'oc20_H_adsorbate_sids.csv'
systemLog = os.path.join(systemLogPath, systemLogName)
sidLog = os.path.join(systemLogPath, sidLogName)
systemLogDF = pd.read_csv(systemLog, header=None, delimiter='^')
systemLogDF.columns = ['System']
sidLogDF = pd.read_csv(sidLog, header=None)
sidLogDF.columns = ['sid']
systemLogDF = pd.concat([sidLogDF['sid'], systemLogDF['System']], axis=1, ignore_index=True)
systemLogDF.columns = ['sid', 'System']
systemDescriptionPD = systemLogDF
systemDescriptionPD['sid'] = systemDescriptionPD['sid'].str.strip('random')

predictionFileList = []
uncertaintyFileList = []

for ind in range(len(predictionFilePathList)):
    predictionFilePath = os.path.join(predictionFilePathList[ind], predictionFileNameList[ind])
    predictionFileList.append(predictionFilePath)
    uncertaintyFilePath = os.path.join(
        uncertaintyFilePathList[ind], uncertaintyFileNameList[ind]
    )
    uncertaintyFileList.append(uncertaintyFilePath)

print('Prediction Files:')
print(predictionFileList)

predictionLabels = []
uncertaintyLabels = []

predictionColNameEnsemble = 'Mean Sample Energy (eV)'
predictionColNameDropout = predictionColNameEnsemble
predictionColNameEvidential = 'Ads. Energy. (eV)'
predictionLabels.append(predictionColNameEnsemble)
predictionLabels.append(predictionColNameDropout)
predictionLabels.append(predictionColNameEvidential)

uncertaintyColNameEnsemble = 'Std. Dev. Sample Energy (eV)'
uncertaintyColNameDropout = uncertaintyColNameEnsemble
epistemicUncertaintyColNameEvidential = 'Epistemic Uncertainty (std. dev., eV)'
uncertaintyLabels.append(uncertaintyColNameEnsemble)
uncertaintyLabels.append(uncertaintyColNameDropout)
uncertaintyLabels.append(epistemicUncertaintyColNameEvidential)

predictionPDList = []

for ind in range(len(predictionFileList)):
    predictionPD = pd.read_csv(predictionFileList[ind])[predictionLabels[ind]]
    uncertaintyPD = pd.read_csv(uncertaintyFileList[ind])[uncertaintyLabels[ind]]
    sidPD = pd.read_csv(predictionFileList[ind])['sid']
    combinedPD = pd.concat([sidPD, predictionPD, uncertaintyPD], axis=1, ignore_index=True)
    combinedPD.columns = ['sid', predictionLabels[ind], uncertaintyLabels[ind]]
    predictionPDList.append(combinedPD)

for ind in range(len(predictionPDList)):
    relevantSystems = apply_heuristic_subselection(
        predictionPDList[ind],
        targetEnergiesSID,
        -0.1,
        0.1,
        0.05,
        y_pred_Label=predictionLabels[ind],
        y_std_Label=uncertaintyLabels[ind],
        sid_Label='sid',
    )
    relevantSystems = relevantSystems.reset_index(drop=True)

    screenedSIDs = list(relevantSystems['sid'].to_numpy())
    screenedSIDs = [str(sid) for sid in screenedSIDs]
    screenedTargets = retrieve_system_info(targetEnergiesPD, screenedSIDs, sid_Label='sid')
    print(screenedTargets)
    screenedTargets = pd.DataFrame(
        screenedTargets['Target (eV)'].to_numpy(), columns=['Target (eV)']
    )

    relevantSystems = pd.concat([relevantSystems, screenedTargets], axis=1, ignore_index=True)
    relevantSystems.columns = [
        'sid',
        predictionLabels[ind],
        uncertaintyLabels[ind],
        'Target (eV)',
    ]
    print(relevantSystems)

    relevantSystemDescriptions = retrieve_system_info(
        systemDescriptionPD, screenedSIDs, sid_Label='sid'
    )

    saveFileName = 'SUBSELECTED_' + uncertaintyFileNameList[ind]
    saveSystemDescriptionName = 'SUBSELECTED_DESC_' + uncertaintyFileNameList[ind]

    saveFile = os.path.join(predictionFilePathList[ind], saveFileName)
    saveDescriptionFile = os.path.join(predictionFilePathList[ind], saveSystemDescriptionName)

    relevantSystems.to_csv(saveFile, header=True, index=False)
    saveFile = os.path.join(uncertaintyFilePathList[ind], saveFileName)
    relevantSystems.to_csv(saveFile, header=True, index=False)
    relevantSystemDescriptions.to_csv(saveDescriptionFile, header=True, index=False)
