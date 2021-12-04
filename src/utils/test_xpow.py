# -*- coding: utf-8 -*-

import numpy as np
from .xpow import xpow
from .read_cif import read_cif

cif_file_path = "cifs/quartz.cif"
# cif_file_path = "cifs/standard_SIO2.cif"
# cif_file_path = "cifs/Portlandite.cif"
# cif_file_path = "cifs/Brownmillerite_0003433.cif"  # perfect, with occupancies
cif_file_path = "cifs/Actinolite_0001983.cif" # RIR low, peak intensities wrong
# cif_file_path = "cifs/Actinolite_0001983_no_temp.cif" # RIR low, peak intensities wrong
# cif_file_path = "cifs/Actinolite_0001983_no_temp_no_occ.cif" # RIR low, peak intensities wrong
# cif_file_path = "cifs/Copper_0011145.cif"

wavelength = 1.790290
wavelength = 1.541838

cif_data = read_cif(cif_file_path)

# default: Cu radiation, 2-theta from 0 to 90, intensity cutoff = 1.0
error_msg, df = xpow(cif_data,wavelength=wavelength)

# Mo radiation:
#error_msg, df = xpow(cif_data,wavelength=0.710730)

# Co radiation:
#error_msg, df = xpow(cif_data,wavelength=3.359480)

# Cu radiation, 2-theta from 20 to 85, show all peaks
#error_msg, df = xpow(cif_data,stt=20,ett=85,tol=0.0001)

if(error_msg != "none"):
    print(error_msg)
else:
    ttheta,intensity,dspacing,hkl,density,MAIPV2,RIR,volume = df
    print(f"x-ray wavelength:  {wavelength}")
    print(f"cell volume:  {volume}")
    print(f"density (g/cm^3):  {density}")
    print(f"max absolute intensity per volume^2:  {MAIPV2}")
    print(f"RIR:  {RIR}")
    print("2-theta  intensity  d-spacing  h  k  l")
    for i in range(len(df[1])):
        print(" %5.2f    %6.2f     %6.4f   %2i %2i %2i" % (ttheta[i],intensity[i],dspacing[i],hkl[i][0],hkl[i][1],hkl[i][2]))
