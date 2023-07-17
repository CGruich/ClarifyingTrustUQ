import joblib
import os.path
import lmdb
from ocpmodels.datasets import LmdbDataset
import csv

# System meta data
sysMetaDataPath = 'METADATA_PATH'
sysMetaDataFile = 'oc20_data_mapping.pkl'
sysMetaData = os.path.join(sysMetaDataPath, sysMetaDataFile)
# Training data
trainPath = 'TRAINING_DATA_PATH'
trainFile = 'data.lmdb'
train = os.path.join(trainPath, trainFile)

# Maximum mapping size for each lmdb file that is to hold an ensemble subsplit
gigSize = 1073741824
numGigs = 20
gigPerLMDB = gigSize * numGigs

train_H_Adsorbate_Path = sysMetaDataPath
train_H_Adsorbate_File = 'Train_H_Adsorbate_data.lmdb'
train_H_Adsorbate = os.path.join(train_H_Adsorbate_Path, train_H_Adsorbate_File)
train_H_Adsorbate_sid_File = 'Train_H_Adsorbate_sid.csv'
train_H_Adsorbate_sid = os.path.join(train_H_Adsorbate_Path, train_H_Adsorbate_sid_File)
train_H_Adsorbate_Systems_File = 'Train_H_Adsorbate_systems.csv'
train_H_Adsorbate_Systems = os.path.join(
    train_H_Adsorbate_Path, train_H_Adsorbate_Systems_File
)

print('Path for OC20 Meta Data:')
print(sysMetaData)
print('\n')
print('Path to Chosen OC20 Dataset:')
print(train)

sysMetaDataContents = joblib.load(sysMetaData)

# Storing the system IDs that all have hydrogen adsorbates
relevant_sids = []

# Path to save the system IDs of all the relevant adsorbate/alloy systems
subselectPath = sysMetaDataPath
subselectIDFile = 'oc20_H_adsorbate_sids.csv'
subselectSystemFile = 'oc20_H_adsorbate_systems.csv'

subselectID = os.path.join(subselectPath, subselectIDFile)
subselectSystem = os.path.join(subselectPath, subselectSystemFile)


# Generate the sid log of all the relevant adsorbate/alloy systems with the specified adsorbate
with open(subselectSystem, 'w') as subselectSystemLog:
    systemWriter = csv.writer(subselectSystemLog)
    with open(subselectID, 'w') as subselectIDLog:
        IDWriter = csv.writer(subselectIDLog)
        for sid in sysMetaDataContents.keys():
            # Adsorbate ID of hydrogen adsorbate is 1
            if sysMetaDataContents[sid]['ads_id'] == 1:
                print((sid, sysMetaDataContents[sid]))
                IDWriter.writerow([sid])
                systemWriter.writerow([sid, str(sysMetaDataContents[sid])])

                relevant_sids.append(int(sid.replace('random', "")))

# Sort the sids of the relevant systems
relevant_sids.sort()
relevant_sids = [str(sid) for sid in relevant_sids]

train_env = lmdb.open(
    train,
    map_size=gigPerLMDB,
    subdir=False,
    readonly=True,
    lock=False,
    map_async=True,
    readahead=False,
    meminit=False,
)
train_numKeys = train_env.stat()['entries']
train_keyRange = range(train_numKeys)
print('Number of raw entries in Train:')
print(str(train_numKeys))

# Store the sids of the relevant systems inside the Val_ID split
relevant_sids_train = []

# Load the Train IS2RE dataset
train_Dataset = LmdbDataset({'src': train})
print(
    'Number of entries for Train processed as an OC20 LMDB dataset\n(Should be consistent with the number of raw entries)'
)
print(len(train_Dataset))

# Map the row index of systems in the train dataset to the associated sid of the system
# This mapping is sid -> rowIndex
LMDB_ind_to_train_sid_mapping = {}
for ind in range(len(train_Dataset)):
    indKey = str(ind)
    sidValue = str(train_Dataset[ind].sid)
    LMDB_ind_to_train_sid_mapping[sidValue] = indKey

# Open the Train dataset with the LMDB tool
with train_env.begin() as txn_old:
    # Open a new LMDB file that we can write sub-selected adsorbate/alloy systems to.
    cop = lmdb.open(
        train_H_Adsorbate,
        map_size=gigPerLMDB,
        subdir=False,
        readonly=False,
        map_async=True,
        lock=True,
        readahead=False,
        meminit=False,
    )

    # With the new LMDB file open,
    with cop.begin(write=True) as txn_new:
        counter = 0

        # For each relevant sid,
        for sidInd in range(len(relevant_sids)):
            # Get the relevant sid,
            sidVal = relevant_sids[sidInd]
            # Get the Train row_ind associated with the sid via the dictionary mapping
            try:
                train_ind = LMDB_ind_to_train_sid_mapping[sidVal].encode('ascii')
            except:
                continue

            # Get the relevant adsorbate/alloy system based on the Train row_ind
            relevantSystem = txn_old.get(key=train_ind)

            # If the relevant adsorbate/alloy system is inside the Train, copy the system
            # over to our new sub-selected LMDB file of only the relevant adsorbate/alloy systems
            if relevantSystem != None:
                # Set up the keys of this new LDMB file as an index ordering of 0, 1, 2 + ... + N
                # We encode the keys in ascii format as standard practice.
                newInd = str(counter).encode('ascii')
                txn_new.put(newInd, relevantSystem)
                print('(New_Index, sid)' + str((str(counter), sidVal)))
                counter += 1
                relevant_sids_train.append(sidVal)

    print('Sub-selection of Systems Completed. Description:')
    print(str(cop.stat()) + '\n')

    # Explicitly ensure we close the new LMDB file
    cop.close()

# Explicitly ensure we close the original LMDB file
train_env.close()

with open(train_H_Adsorbate_sid, 'w') as subselectIDLog:
    for sid in relevant_sids_train:
        subselectIDLog.write(sid)
        subselectIDLog.write('\n')

with open(train_H_Adsorbate_Systems, 'w') as subselectDescLog:
    systemWriter = csv.writer(subselectDescLog)
    for sid in relevant_sids_train:
        sid = 'random' + str(sid)
        systemWriter.writerow([sid.replace('random', ""), str(sysMetaDataContents[sid])])
