"""
Main

Downloads a full matched pdb in 'config defined' folder and prints the accession of structure.

Edit this file at your convenience

"""
from modules.get_pdb import FindPdb
from modules import config
import pandas as pd
import os
import sys
import glob

data_path = sys.argv[1]
base_name = sys.argv[2]
export_path = sys.argv[3]

def main():
    input_df = pd.read_csv(f"{data_path}{base_name}.csv")
    total = len(input_df)
    if os.path.exists(f"{config.TEMP_FOLDER}/{base_name}.csv"):
        processed_data = pd.read_csv(f"{config.TEMP_FOLDER}/{base_name}.csv")
    else:
        processed_data = pd.DataFrame(columns=['index','PDB_ID'])

    for row in input_df.itertuples():
        if row.index in processed_data['index'].values:
            continue

        if (row.Index + 1) % 100 == 0:
            processed_data.to_csv(f"{config.TEMP_FOLDER}/{base_name}.csv", index=False)

        print(f"( {row.Index} / {total} ) Searching: {row.organism}:{row.sequence_id}\t{row.sequence_aa[:15]}...", end='\t')
        find_pdb = FindPdb(row.sequence_aa)
        pdb_accession = find_pdb.get_pdb()
            
        if pdb_accession:
            #input_df.loc[row.Index, 'PDB_ID'] = pdb_accession
            processed_data.loc[len(processed_data.index)] = [row.index, pdb_accession]
            print("OK")
        else:
            processed_data.loc[len(processed_data.index)] = [row.index, None]
            print("")

    processed_data = processed_data.drop(columns="index")
    input_df = pd.concat([input_df, processed_data], axis=1)
    input_df.to_csv(f"{export_path}{base_name}.csv", index=False)

if __name__ == "__main__":
    main()
