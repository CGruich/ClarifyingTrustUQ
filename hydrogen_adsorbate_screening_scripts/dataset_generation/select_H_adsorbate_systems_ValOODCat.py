import joblib
import os.path
import lmdb
from ocpmodels.datasets import LmdbDataset
import csv

sysMetaDataPath = "METADATA_PATH"
sysMetaDataFile = "oc20_data_mapping.pkl"
sysMetaData = os.path.join(sysMetaDataPath, sysMetaDataFile)
valOOD_CatPath = "VALOOD_PATH"
valOOD_CatFile = "data.lmdb"
valOOD_Cat = os.path.join(valOOD_CatPath, valOOD_CatFile)

# Maximum mapping size for each lmdb file that is to hold an ensemble subsplit
gigSize = 1073741824
numGigs = 20
gigPerLMDB = gigSize * numGigs

valOOD_Cat_H_Adsorbate_Path = sysMetaDataPath
valOOD_Cat_H_Adsorbate_File = "ValOOD_Cat_H_Adsorbate_data.lmdb"
valOOD_Cat_H_Adsorbate = os.path.join(valOOD_Cat_H_Adsorbate_Path, valOOD_Cat_H_Adsorbate_File)
valOOD_Cat_H_Adsorbate_sid_File = "ValOOD_Cat_H_Adsorbate_sid.csv"
valOOD_Cat_H_Adsorbate_sid = os.path.join(valOOD_Cat_H_Adsorbate_Path, valOOD_Cat_H_Adsorbate_sid_File)
valOOD_Cat_H_Adsorbate_Systems_File = "ValOOD_Cat_H_Adsorbate_systems.csv"
valOOD_Cat_H_Adsorbate_Systems = os.path.join(valOOD_Cat_H_Adsorbate_Path, valOOD_Cat_H_Adsorbate_Systems_File)

print("Path for OC20 Meta Data:")
print(sysMetaData)
print("\n")
print("Path to Chosen OC20 Dataset:")
print(valOOD_Cat)

sysMetaDataContents = joblib.load(sysMetaData)

# Storing the system IDs that all have hydrogen adsorbates
relevant_sids = []

# Path to save the system IDs of all the relevant adsorbate/alloy systems
subselectPath = sysMetaDataPath
subselectIDFile = "oc20_H_adsorbate_sids.csv"
subselectSystemFile = "oc20_H_adsorbate_systems.csv"

subselectID = os.path.join(subselectPath, subselectIDFile)
subselectSystem = os.path.join(subselectPath, subselectSystemFile)

# Generate the sid log of all the relevant adsorbate/alloy systems with the specified adsorbate
with open(subselectSystem, "w") as subselectSystemLog:
    with open(subselectID, "w") as subselectIDLog:
        for sid in sysMetaDataContents.keys():
            # Adsorbate ID of hydrogen adsorbate is 1
            if sysMetaDataContents[sid]["ads_id"] == 1:
                print((sid, sysMetaDataContents[sid]))
                subselectIDLog.write(sid)
                subselectIDLog.write("\n")
                subselectSystemLog.write(str(sysMetaDataContents[sid]))
                subselectSystemLog.write("\n")

                relevant_sids.append(int(sid.replace("random", "")))

# Sort the sids of the relevant systems
relevant_sids.sort()
relevant_sids = [str(sid) for sid in relevant_sids]

valOOD_Cat_env = lmdb.open(valOOD_Cat, map_size=gigPerLMDB, subdir=False, readonly=True, lock=False, map_async=True, readahead=False, meminit=False)
valOOD_Cat_numKeys = valOOD_Cat_env.stat()["entries"]
valOOD_Cat_keyRange = range(valOOD_Cat_numKeys)
print("Number of raw entries in dataset:")
print(str(valOOD_Cat_numKeys))

# Store the sids of the relevant systems inside the Val_ID split
relevant_sids_valOOD_Cat = []

# Load the ValID IS2RE dataset
valOOD_Cat_Dataset = LmdbDataset({"src": valOOD_Cat})
print("Number of entries for dataset processed as an OC20 LMDB dataset\n(Should be consistent with the number of raw entries)")
print(len(valOOD_Cat_Dataset))

# Map the row index of systems in the valOOD_Cat dataset to the associated sid of the system
# This mapping is sid -> rowIndex
LMDB_ind_to_valOOD_Cat_sid_mapping = {}
for ind in range(len(valOOD_Cat_Dataset)):
    indKey = str(ind)
    sidValue = str(valOOD_Cat_Dataset[ind].sid)
    LMDB_ind_to_valOOD_Cat_sid_mapping[sidValue]=indKey

# Open the ValID dataset with the LMDB tool
with valOOD_Cat_env.begin() as txn_old:
    # Open a new LMDB file that we can write sub-selected adsorbate/alloy systems to.
    cop = lmdb.open(valOOD_Cat_H_Adsorbate, map_size=gigPerLMDB, subdir=False, readonly=False, map_async=True, lock=True, readahead=False, meminit=False)
    
    # With the new LMDB file open,
    with cop.begin(write=True) as txn_new:
        counter = 0
        
        # For each relevant sid,
        for sidInd in range(len(relevant_sids)):
            # Get the relevant sid,
            sidVal = relevant_sids[sidInd]
            # Get the ValID row_ind associated with the sid via the dictionary mapping
            try:
                valOOD_Cat_ind = LMDB_ind_to_valOOD_Cat_sid_mapping[sidVal].encode("ascii")
            except:
                continue
            
            # Get the relevant adsorbate/alloy system based on the ValID row_ind
            relevantSystem = txn_old.get(key=valOOD_Cat_ind)
            
            # If the relevant adsorbate/alloy system is inside the Val_ID, copy the system
            # over to our new sub-selected LMDB file of only the relevant adsorbate/alloy systems
            if relevantSystem != None:
                # Set up the keys of this new LDMB file as an index ordering of 0, 1, 2 + ... + N
                # We encode the keys in ascii format as standard practice.
                newInd = str(counter).encode("ascii")
                txn_new.put(newInd, relevantSystem)
                print("(New_Index, sid)" + str((str(counter), sidVal)))
                counter += 1
                relevant_sids_valOOD_Cat.append(sidVal)
    
    print("Sub-selection of Systems Completed. Description:")
    print(str(cop.stat()) + "\n")

    # Explicitly ensure we close the new LMDB file
    cop.close()

# Explicitly ensure we close the original LMDB file
valOOD_Cat_env.close()

with open(valOOD_Cat_H_Adsorbate_sid, "w") as subselectIDLog:
    for sid in relevant_sids_valOOD_Cat:
        subselectIDLog.write(sid)
        subselectIDLog.write("\n")

with open(valOOD_Cat_H_Adsorbate_Systems, "w") as subselectDescLog:
    systemWriter = csv.writer(subselectDescLog)
    for sid in relevant_sids_valOOD_Cat:
        sid = "random" + str(sid)
        systemWriter.writerow([sid.replace("random", ""), str(sysMetaDataContents[sid])])
        
