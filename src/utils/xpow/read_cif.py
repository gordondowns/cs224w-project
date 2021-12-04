# -*- coding: utf-8 -*-

def read_cif(cif_file_path) :
    import numpy as np
    from . import symmmaker #wrapped fortran code for translating string matrix descriptions to matrices
    import CifFile #cif reader from https://pypi.python.org/pypi/PyCifRW/4.2.1
    
    mycif = CifFile.CifFile()
    myblock = CifFile.CifBlock()
    mycif['a_block'] = myblock
    mycif = CifFile.ReadCif(cif_file_path,grammar='auto')
    data = mycif.first_block()
    
    num_sym = len(data['_space_group_symop_operation_xyz'])
    symopmatrices = []
    for i in range(num_sym):
         symopmatrices.append(np.asfortranarray((symmmaker.make(data['_space_group_symop_operation_xyz'][i]))))
    sym_array = np.zeros([num_sym,3,4])
    for i in range(num_sym):
        sym_array[i,:,:] = np.array(symopmatrices[i][0])
    
    atom_names = np.array(data["_atom_site_label"])
    x_array = np.array(data["_atom_site_fract_x"]).astype(float)
    y_array = np.array(data["_atom_site_fract_y"]).astype(float)
    z_array = np.array(data["_atom_site_fract_z"]).astype(float)
    atom_xyz = np.matrix(data=(x_array,y_array,z_array)).transpose()
    cell_params = np.array([data['_cell_length_a'],data['_cell_length_b'],
                            data['_cell_length_c'],data['_cell_angle_alpha'],
                            data['_cell_angle_beta'],data['_cell_angle_gamma']
                            ] ).astype(float)
    num_atoms = len(atom_names)

    # there may or may not be occupancies in the CIF.
    try:
        atom_occ = np.array(data["_atom_site_occupancy"]).astype(float)
    except:
        atom_occ = np.ones(num_atoms).astype(float)
    
    # There may or may not be temperature factors in the CIF.
    # If there are, they may be of mixed types. Convert them all to Bisos.
    # XPow automatically assigns Biso values to atoms with Bisos of 0.0
    atom_Biso = np.zeros(num_atoms).astype(float)
    for i in range(num_atoms):
        try:
            atom_Biso[i] = float(data["_atom_site_B_iso_or_equiv"][i])
            # this will correctly throw an exception if Biso is "?"
        except:
            try:
                atom_Biso[i] = float(data["_atom_site_U_iso_or_equiv"][i]) * (np.pi**2)*8.0
            except:
                try:
                    B11 = float(data["_atom_site_aniso_B_11"][i])
                    B22 = float(data["_atom_site_aniso_B_22"][i])
                    B33 = float(data["_atom_site_aniso_B_33"][i])
                    B12 = float(data["_atom_site_aniso_B_12"][i])
                    B13 = float(data["_atom_site_aniso_B_13"][i])
                    B23 = float(data["_atom_site_aniso_B_23"][i])
                    a = cell_params[0]
                    b = cell_params[1]
                    c = cell_params[2]
                    A = np.deg2rad(cell_params[3]) #alpha
                    B = np.deg2rad(cell_params[4]) #beta
                    C = np.deg2rad(cell_params[5]) #gamma
                    G = np.matrix([ [a*a, a*b*np.cos(C), a*c*np.cos(B)],
                                    [a*b*np.cos(C), b*b, b*c*np.cos(A)],
                                    [a*c*np.cos(B), b*c*np.cos(A), c*c] ])
                    Beta = np.matrix([ [B11, B12, B13],
                                       [B12, B22, B23],
                                       [B13, B23, B33] ])
                    atom_Biso[i] = (np.trace(Beta*G)) * 4/3
                except:
                    try:
                        U11 = float(data["_atom_site_aniso_U_11"][i])
                        U22 = float(data["_atom_site_aniso_U_22"][i])
                        U33 = float(data["_atom_site_aniso_U_33"][i])
                        U12 = float(data["_atom_site_aniso_U_12"][i])
                        U13 = float(data["_atom_site_aniso_U_13"][i])
                        U23 = float(data["_atom_site_aniso_U_23"][i])
                        a = cell_params[0]
                        b = cell_params[1]
                        c = cell_params[2]
                        A = np.deg2rad(cell_params[3]) #alpha
                        B = np.deg2rad(cell_params[4]) #beta
                        C = np.deg2rad(cell_params[5]) #gamma
                        G = np.matrix([ [a*a, a*b*np.cos(C), a*c*np.cos(B)],
                                        [a*b*np.cos(C), b*b, b*c*np.cos(A)],
                                        [a*c*np.cos(B), b*c*np.cos(A), c*c] ])
                        D = np.matrix([ [np.sqrt(G.I[0,0]), 0.0, 0.0],
                                        [0.0, np.sqrt(G.I[1,1]), 0.0],
                                        [0.0, 0.0, np.sqrt(G.I[2,2])] ])
                        U = np.matrix([ [U11, U12, U13],
                                        [U12, U22, U23],
                                        [U13, U23, U33] ])
                        atom_Biso[i] = (np.trace(D*U*D*G)) * (((np.pi)**2)*8/3)
                    except:
                        atom_Biso[i] = 0.0
                        # XPow will automatically assign a reasonable value
    
    # negative Biso means data is garbage. Discard that reported Biso.
    for i in range(num_atoms):
        if(atom_Biso[i] < 0):
            atom_Biso[i] = 0.0
    
    # eliminate everything after the atom's chemical symbol.
    good_atom_names = ['He','Li','Be','Ne','Na','Mg','Al','Si','Cl','Ar',
    'Ca','Sc','Ti','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se',
    'Br','Kr','Rb','Sr','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In',
    'Sn','Sb','Te','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd',
    'Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','Re','Os','Ir','Pt','Au',
    'Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th','Pa','Np','Pu',
    'Am','Cm','Bk','Cf','Wa','H','B','C','N','O','F','P','S','K','V','Y',
    'I','W','U','D'] # reordered to prevent false positives
    for i in range(num_atoms):
        for j in range(len(good_atom_names)):
            if(atom_names[i][0:len(good_atom_names[j])] == good_atom_names[j]):                
                atom_names[i] = good_atom_names[j]
                break
    
    output = (cell_params,sym_array,atom_names,atom_xyz,atom_occ,atom_Biso)
    
    return output