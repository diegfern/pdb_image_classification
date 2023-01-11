from .utils_functions import utils_functions
import pandas as pd
import numpy as np

class process_pdb_files(object):

    def __init__(
        self, 
        pdb_doc=None, 
        code_pdb=None,
        type_document='PDB'):

        self.pdb_doc = pdb_doc
        self.code_pdb = code_pdb
        self.type_document = type_document

        self.utils_instance = utils_functions()

    def parsing_pdb_files(
        self):

        self.structure = self.utils_instance.pdb_parser(
                code_pdb=self.code_pdb, 
                name_pdb=self.pdb_doc, 
                type_document=self.type_document)

        #get all atoms
        self.list_atoms = self.structure.get_atoms()

    def atoms_to_df(self):

        matrix_data = []

        for atom in self.list_atoms:

            residue = atom.get_parent()
            chain = residue.get_parent()

            residue_id = residue.get_id()
            chain_id = chain.get_id()
            row = [
                chain_id,
                residue_id[0],
                residue_id[1],
                residue_id[2],
                residue.resname,
                atom.get_name(),
                atom.get_id()]

            coord = atom.get_coord()
            row.append(coord[0])
            row.append(coord[1])
            row.append(coord[2])
            
            matrix_data.append(row)            

        self.df_atoms = pd.DataFrame(matrix_data, columns=["chain", "residue-hetero-flag", "residue-sequence-identifier", "residue-insertion-code", "resname", "name_atom", "id_atom", "X", "Y", "Z"])


    def structure_to_image(
        self, 
        ignore_het_atom=False,
        zero_padding=None,
        name_export=None,
        size_reshape=16):

        df_data = self.df_atoms[["residue-hetero-flag", "X", "Y", "Z"]]

        if ignore_het_atom == True:
            df_data = df_data.dropna()
        
        df_data = df_data[["X", "Y", "Z"]]

        data_values = df_data.values

        if zero_padding != None:
            zero_matrix = np.zeros((zero_padding, 3))
            data_values = np.concatenate((data_values, zero_matrix), axis=0)
        
        data_values = np.array(data_values, dtype=np.uint8)
        data_values = np.resize(data_values, (size_reshape, size_reshape))
        
        #create the figure
        self.utils_instance.create_figure_from_np_array(
            data_matrix=data_values,
            name_export=name_export
        )
