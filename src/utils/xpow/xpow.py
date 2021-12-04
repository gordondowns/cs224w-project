def xpow(data,wavelength=1.541838,stt=0.0,ett=90.0,tol=1.0):
    #makes it easy to run the xpow module named xpowf
    
    import numpy as np
    from .xpowf import xpowf
    
    cell_params = data[0] #cell parameters: a,b,c,alpha,beta,gamma
    sym_array = data[1]   #array of 3x4 symmetry matrices
    atom_names = data[2]  #array of 2-character symbols of atoms
    atom_xyz = data[3]    #matrix of xyz atomic coordinates
    atom_occ = data[4]    #array of atom occupancies
    atom_Biso = data[5]   #array of atom Beta isotropic temperature factors
    num_atoms = len(atom_names)  #number of atoms in the dataset

    # need to break atom_names into two one-character arrays
    # because f2py could not convert python strings correctly
    atom_n1 = []
    atom_n2 = []
    for i in range(num_atoms):
        atom_n1.append(atom_names[i][0])
        if(len(atom_names[i]) == 2):
            atom_n2.append(atom_names[i][1])
        else:
            atom_n2.append(" ")
    atom_n1 = np.array(atom_n1)
    atom_n2 = np.array(atom_n2)
    
    flag,outtt,outtint,outds,outhkl,num_data,density,MAIPV2,RIR,volume = xpowf(cell_params,
        atom_n1,atom_n2,atom_xyz,atom_occ,atom_Biso,sym_array,
        wavelength,stt,ett,tol)
    
    if(flag == 0):
        error_msg = "none"
    elif(flag == 1):
        error_msg = "Exceeded the maximum number of reflections, contact authors."
    elif(flag == 17):
        error_msg = "One of the atom names is not recognized. Use chemical symbol."
    elif(flag == 18):
        error_msg = "Neutron scattering factors not available for atom in datafile."
    elif(flag == 27):
        error_msg = "Error interpreting the cell parameters."
    else:
        error_msg = "Error could not be identified." #this should never happen
    
    output = [ outtt[0:num_data],outtint[0:num_data],
               outds[0:num_data],outhkl[0:num_data,:],density,MAIPV2,RIR,volume ]
    
    # outtt = array of 2-thetas that have peaks
    # outtint = array of peak intensities at those 2-thetas
    # outds = array of corresponding d-spacings
    # outhkl = array of corresponding hkls
    # MAIPV2 = Maximum Absolute Intensity Per Volume Squared
    # RIR    = Reference Intensity Ratio
    
    return error_msg, output
    
def MakeGaussiansFromPeaks(peaksXY,sigma):
    #turns the peaks into full xy data for plotting
    
    import numpy as np
    
    xin = peaksXY[0]
    yin = peaksXY[1]

    xmin = 0.0
    xmax = 90.0
    step = 0.05
    
    xout = np.arange(xmin, xmax+step, step)
    
    yout = np.zeros_like(xout)
    
    for i in range(0,len(xout)):
        for j in range(0,len(xin)):
            yout[i] += yin[j] * ( 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (xout[i] - xin[j])**2 / (2 * sigma**2)) )
            # there are certainly faster methods of doing this
    
    return [xout,yout]