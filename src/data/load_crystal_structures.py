from ..utils.xpow.read_cif import read_cif
# from read_cif import read_cif
import numpy as np
from numpy import cos, sin
from glob import glob
import networkx as nx
import pandas as pd


# get dicts from periodic table
df = pd.read_csv("../data/raw/periodic_table_from_wikipedia.csv")
atomic_number_dict = dict(zip(df['Symbol'], df["Atomic number"]))
atomic_weight_dict = dict(zip(df['Symbol'], df["Standard atomic weight"]))
electronegativity_dict = dict(zip(df['Symbol'], df["Electronegativity"]))


def direct_coords_to_cartesian_coords(atom_xyz_direct,cell_params):
    # convert to cartesian space using formula from https://en.wikipedia.org/wiki/Fractional_coordinates
    a,b,c,alpha,beta,gamma = cell_params
    alpha,beta,gamma = np.deg2rad(alpha), np.deg2rad(beta), np.deg2rad(gamma)
    omega = a*b*c*np.sqrt(1. - cos(alpha)*cos(alpha) - cos(beta)*cos(beta) - cos(gamma)*cos(gamma) \
                            + 2*cos(alpha)*cos(beta)*cos(gamma))
    direct_to_cartesian_transform_matrix = np.array([
        [a, b*cos(gamma), c*cos(beta)],
        [0, b*sin(gamma), c*(cos(alpha)-cos(beta)*cos(gamma))/sin(gamma)],
        [0, 0, omega/(a*b*sin(gamma))]
    ])
    atom_xyz_cartesian = np.array(atom_xyz_direct) @ direct_to_cartesian_transform_matrix.T
    return atom_xyz_cartesian


def load_single_crystal_structure(file_path, min_distance_for_edge=0.001, max_distance_for_edge=np.inf, verbose=False):
    
    mineral_name = file_path.split('\\')[-1].split('/')[-1].split('0')[0]
    out = read_cif(file_path)
    cell_params,atom_elements,atom_xyz,atom_occ,atom_Biso = out
    if verbose:
        print("mineral name:",mineral_name)
        print("cell parameters:",cell_params)

    atom_xyz_cartesian = direct_coords_to_cartesian_coords(atom_xyz,cell_params)
    G = nx.Graph()
    if verbose:
        print("")
        print("i el atomic_number atomic_weight electronegativity xyz occ Biso")
    for i,(el,xyz,occ,Biso) in enumerate(zip(atom_elements,atom_xyz_cartesian,atom_occ,atom_Biso)):
        # look up atomic number and weight
        atomic_number= atomic_number_dict[el]
        atomic_weight = atomic_weight_dict[el]
        electronegativity = electronegativity_dict[el]
        G.add_node(i,
            element = el, 
            xyz = xyz, 
            occ = occ, 
            Biso = Biso, 
            atomic_number = atomic_number, 
            atomic_weight = atomic_weight, 
            electronegativity = electronegativity,
        )
        if verbose:
            print(i,el,atomic_number,atomic_weight,xyz,electronegativity,occ,Biso)

    for i1,xyz1 in enumerate(atom_xyz_cartesian[:-1]):
        for i2,xyz2 in enumerate(atom_xyz_cartesian[i1+1:],start=i1+1):
            dist = np.linalg.norm(xyz1-xyz2)
            if min_distance_for_edge < dist < max_distance_for_edge:
                G.add_edge(i1,i2,dist=dist)

    if verbose:
        drawing_labels = nx.get_node_attributes(G,'element')
        nx.draw(G,labels=drawing_labels)

    return mineral_name, G


def load_cif_data(
        CIF_directory_path='../data/raw/CIFs/',
        verbose=False):
    file_paths_list = glob(CIF_directory_path+'*.cif')
    mineral_names = []
    graphs = []
    for fp in file_paths_list:
        mineral_name, graph = load_single_crystal_structure(fp, verbose=verbose)
        mineral_names.append(mineral_name)
        graphs.append(graph)
    return file_paths_list, mineral_names, graphs

