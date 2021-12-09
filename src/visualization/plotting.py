from matplotlib import pyplot as plt
import numpy as np

def plot_one_dataset(intensity,tt=None):
    plt.figure()
    if tt is None:
        plt.plot(intensity)
    else:
        plt.plot(tt,intensity)
        plt.xlim((tt[0],tt[-1]))
    plt.grid()
    plt.show()

def plot_spectra(pts_pred, pts_true=None, legend=True):
    plots = []
    xs_pred = [pt[0] for pt in pts_pred]
    ys_pred = [pt[1] for pt in pts_pred]

    plots.append(plt.plot(xs_pred, ys_pred, label='Predicted spectrum')[0])
    
    if pts_true is not None:
        xs_true = [pt[0] for pt in pts_true]
        ys_true = [pt[1] for pt in pts_true]
        plots.append(plt.plot(xs_true, ys_true, label='True spectrum')[0])

    if legend:
        plt.legend(handles=plots)
    plt.ylabel('Intensity')
    plt.xlabel('Raman shift (cm^-1)')
    plt.show()
