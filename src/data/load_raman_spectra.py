from ..utils import qxrd_apc as apc
import numpy as np
from glob import glob


def load_raman_data(
        model_wavenumber_values, #np.load(repo_file_paths.model_wavenumber_paths[version])
        raman_data_directory_path='../data/raw/raman/',
        wavelength=None,
        verbose=False):

    if wavelength:
        file_paths_list = glob(raman_data_directory_path+f'*/*__{wavelength}__*.txt')
    else:
        file_paths_list = glob(raman_data_directory_path+'*/*.txt')
    mineral_names = []
    raman_spectra = []
    wavelengths = []
    for fp in file_paths_list:
        try:
            mineral_name, raman_spectrum, wavelength = load_single_raman_spectrum(model_wavenumber_values,fp)
            mineral_names.append(mineral_name)
            raman_spectra.append(raman_spectrum)
            wavelengths.append(wavelength)
        except Exception as e:
            if verbose:
                print(e)
                print(f"problem file: {fp}")
                print("")
    
    # raman_spectra = np.vstack(raman_spectra)
    return file_paths_list, mineral_names, wavelengths, raman_spectra


def load_single_raman_spectrum(
        model_wavenumber_values, #np.load(repo_file_paths.model_wavenumber_paths[version])
        file_path):
    mineral_name = file_path.split('\\')[-1].split('/')[-1].split('__')[0]
    wavelength = int(file_path.split('__Raman__')[-1].split('__')[0])
    temp_apc = apc.TopLevel(file_path,twotheta_ranges=[(0.0,100000.0)],print_warnings=False)
    if 0 in temp_apc.input_profile.xy_data[1]:
        raise Exception('Model wavenumbers too broad for this spectrum. Skip.')
    raman_spectrum = process_raman_spectrum(temp_apc.input_profile.xy_data,model_wavenumber_values)
    return mineral_name, wavelength, raman_spectrum


def process_raman_spectrum(xy,model_twotheta_values,zero_pad=True):
    if zero_pad:
        intensity_interpolated = np.interp(model_twotheta_values,xy[0],xy[1],left=0.0,right=0.0)
    else:
        intensity_interpolated = np.interp(model_twotheta_values,xy[0],xy[1])
    intensity_normalized = np.multiply(intensity_interpolated,1.0/np.max(intensity_interpolated))
    return intensity_normalized
