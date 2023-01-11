"""
Match sequences and download pdb
"""
import requests
from modules import config
from modules.blastp import Blast
import os
import time

class FindPdb:
    """Match sequences against db class and gets PDB file from alphafold / rcsb pdb"""
    def __init__(self, sequence):
        self.sequence = sequence
        self.db_names = ["swissprot", "pdbaa"]
        self.alphafold_path = """https://alphafold.ebi.ac.uk/files/AF-{}-F1-model_v4.pdb"""
        self.rscb_path = """https://files.rcsb.org/download/{}.pdb"""

    def __get_match(self):
        """execute process"""
        complete_list = []
        for db_name in self.db_names:
            blastp = Blast(self.sequence, db_name)
            try:
                swissprot_response = blastp.run_process()["data"]
            except KeyError:
                continue
            response = [(a["accession"], a["length"], a["identity"], a["gaps"], a['e_value'], db_name)
                for a in swissprot_response]
            # Obtener el maximo de identity...
            complete_list = complete_list + response
            for res in response:
                accession = res[0]
                length = res[1]
                identity = res[2]
                gaps = res[3]
                if gaps == 0 and identity == length:
                 return db_name, accession

        # Deleting gaps
        complete_list = [x for x in complete_list if x[3] == 0]
        # Sort by e_value
        complete_list = sorted(complete_list, key=lambda tup: tup[4])
        complete_list = complete_list[:5]

        if not complete_list:
            return None
        # Max identity
        max_tuple = max(complete_list, key=lambda tup: tup[2])
        for res in complete_list:
            accession = res[0]
            identity = res[2]
            if identity == max_tuple[2]:
                return res[5], accession

        return None

    def get_pdb(self):
        """Downloads pdb in specified folder"""
        match_result = self.__get_match()
        if match_result is not None:
            db_found, accession = match_result
            if db_found == "swissprot":
                download_path = self.alphafold_path.format(accession)
            elif db_found == "pdbaa":
                download_path = self.rscb_path.format(accession)

            if os.path.exists(f"{config.PDB_FOLDER}/{accession}.pdb"):
                return accession

            for _ in range(3):
                try:
                    res = str(requests.get(download_path, timeout=5000).text)
                except requests.exceptions.ConnectTimeout:
                    time.sleep(10)
                    print("Exception raised: connection Timeout, retrying in 10 seconds...")
                    continue
                break

            with open(f"{config.PDB_FOLDER}/{accession}.pdb", mode="w", encoding="utf-8") as file:
                file.write(res)
            return accession
        return None
