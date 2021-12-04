import numpy as np

def process_raman_spectrum(xy,model_twotheta_values,zero_pad=True):
    if zero_pad:
        intensity_interpolated = np.interp(model_twotheta_values,xy[0],xy[1],left=0.0,right=0.0)
    else:
        intensity_interpolated = np.interp(model_twotheta_values,xy[0],xy[1])
    intensity_normalized = np.multiply(intensity_interpolated,1.0/np.max(intensity_interpolated))
    # intensity_reshaped = intensity_normalized.reshape([1]+list(intensity_normalized.shape)+[1])
    return intensity_normalized
