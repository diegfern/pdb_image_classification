import sys
import os
import warnings
from tqdm import tqdm
import pandas as pd
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from modules.process_pdb import process_pdb_files

responses = ['Oxidoreductases','Transferases','Hydrolases','Lyases','Isomerases','Ligases','Translocases']
data_path = sys.argv[1]
pdb_path = sys.argv[2]
export_path = sys.argv[3]
image_size = int(sys.argv[4])
samples_by_ec = int(sys.argv[5])

warnings.filterwarnings("error")

input_df = pd.read_csv(data_path)
input_df = input_df.dropna()
input_df = input_df.drop_duplicates(subset=["PDB_ID"], keep='first')
input_df = input_df.reset_index(drop=True)

indexes_to_drop = []

with tqdm(total=samples_by_ec*7, desc=f"Getting {image_size}x{image_size} images") as progress_bar:
    for ec in range(1,8):
        ec_p = input_df.loc[input_df['enzyme_code'].str.startswith(str(ec) + '.')]
        img_processed = 0
        for row in ec_p.itertuples(index=True):
            pdb_id = row.PDB_ID
            instance_doc = process_pdb_files(
                code_pdb=pdb_id, 
                pdb_doc="{}{}".format(pdb_path, pdb_id + ".pdb"),
                type_document='PDB')

            try:
                instance_doc.parsing_pdb_files()
            except PDBConstructionWarning:
                indexes_to_drop.append(row.Index)
                continue
            except UnicodeDecodeError:
                indexes_to_drop.append(row.Index)
                continue
            instance_doc.atoms_to_df()

            if not os.path.exists(export_path):
                os.mkdir(export_path)
                for response in responses:
                    os.mkdir(f"{export_path}{response}")

            instance_doc.structure_to_image(
                name_export=f"{export_path}{responses[row.response]}/{pdb_id}.png",
                size_reshape=image_size
                )
            
            progress_bar.update(1)
            img_processed+=1
            if img_processed == samples_by_ec:
                break

#input_df.drop(input_df.index[indexes_to_drop], inplace=True)
#input_df.to_csv(f"{export_path}sequences.csv", index=False)


