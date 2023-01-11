"""Blast module"""
import subprocess
import random
import string
import json
import modules.config as config
import os

class Blast():
    """Create init blast dbs from fasta files"""
    def __init__(self, fasta_text, db):
        self.blast_db = db
        self.fasta_path = self.__create_fasta_file(fasta_text)
        self.out_path = config.TEMP_FOLDER + '/' + ''.join(
            random.choice(string.ascii_letters) for _ in range(10)
            ) + ".out"
        self.ids = []

    def __create_fasta_file(self, fasta_text):
        fasta_path = config.TEMP_FOLDER + '/' + ''.join(
            random.choice(string.ascii_letters) for _ in range(10)
            ) + ".fasta"
        with open(fasta_path, "w", encoding="utf-8") as file:
            file.write(">seq\n")
            file.write(fasta_text)
        return fasta_path

    def __execute_blastp(self):
        command = [
            config.NCBI_BLAST_FOLDER + "/blastp",
            "-db",
            f"{config.BLASTDB_FOLDER}/{self.blast_db}/{self.blast_db}",
            "-query",
            self.fasta_path,
            "-out",
            self.out_path,
            "-outfmt",
            "15"
        ]
        subprocess.check_output(command)

    def __parse_response(self):
        with open(self.out_path, "r", encoding="utf-8") as output_file:
            json_data = json.loads(output_file.read())
        try:
            hits = json_data["BlastOutput2"][0]["report"]["results"]["search"]["hits"]
            data = []
            for hit in hits:
                length = hit["len"]
                for hsp in hit["hsps"]:
                    accession = hit["description"][0]["accession"]
                    identity = hsp["identity"]
                    gaps = hsp["gaps"]
                    similarity = hsp["positive"]
                    row = {"accession": accession}
                    row["bit_score"] = hsp["bit_score"]
                    row["e_value"] = hsp["evalue"]
                    row["length"] = length
                    row["identity"] = identity
                    row["gaps"] = gaps
                    row["similarity"] = similarity
                    data.append(row)
            return data
        except KeyError:
            return None

    def __delete_temp_file(self):
        if os.path.exists(self.fasta_path):
            os.remove(self.fasta_path)
        if os.path.exists(self.out_path):
            os.remove(self.out_path)

    def run_process(self):
        """Runs blastp full process"""
        self.__execute_blastp()
        res = self.__parse_response()
        self.__delete_temp_file()
        if res:
            return {"data": res}
        return {"error": "No significant results"}
