# from ..utils.read_cif import read_cif
from read_cif import read_cif
import numpy as np
from glob import glob
import networkx as nx

def load_single_crystal_structure(file_path):
    
    mineral_name = file_path.split('\\')[-1].split('/')[-1].split('0')[0]
    out = read_cif(file_path)
    cell_params,sym_array,atom_names,atom_xyz,atom_occ,atom_Biso = out
    print(out)

    G = nx.Graph()

    return mineral_name, G
