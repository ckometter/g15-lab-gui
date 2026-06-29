import numpy as np


def get_spectrum_numpy_old(winspec, get_lambdas=True):
    '''Returned array is already summed over frames. Only first ROI is returned.'''
    if get_lambdas is True:
        spect, lambdas = winspec.get_spectrum(wlen=True)
        spect = np.transpose(np.array(spect, dtype=np.uint16),
                             (0, 2, 1))
        lambdas = np.array(lambdas)

        return lambdas, np.sum(spect, axis=0)[0]
    else:
        spect = winspec.get_spectrum(wlen=False)
        spect = np.transpose(np.array(spect, dtype=np.uint16),
                             (0, 2, 1))

        return np.sum(spect, axis=0)[0]


def get_frames_numpy_old(winspec, get_lambdas=True):
    '''Returned array has axes: (frame_nr, ROI, wavelength)'''
    if get_lambdas is True:
        spect, lambdas = winspec.get_spectrum(wlen=True)
        spect = np.transpose(np.array(spect, dtype=np.uint16),
                             (0, 2, 1))
        lambdas = np.array(lambdas)

        return lambdas, spect
    else:
        spect = winspec.get_spectrum(wlen=False)
        spect = np.transpose(np.array(spect, dtype=np.uint16),
                             (0, 2, 1))

        return spect

def get_frames_numpy(winspec, get_lambdas=True):
    #Returned array has axes: (ROI, frame_nr, wavelength)
    if get_lambdas is True:
        spect, lambdas = winspec.get_spectrum(wlen=True)
        spect = np.transpose(np.array(spect, dtype=np.uint16),
                             (2, 0, 1))
        lambdas = np.array(lambdas)

        return lambdas, spect
    else:
        spect = winspec.get_spectrum(wlen=False)
        spect = np.transpose(np.array(spect, dtype=np.uint16),
                             (2, 0, 1))

        return spect
        
def get_frames_numpy_new(winspec, get_lambdas=True):
    '''This is here for compatibility. Will be removed soon.'''
    return get_frames_numpy(winspec, get_lambdas=get_lambdas)


def save_spectrum(fpath, lambdas, spectrum):
    with open(fpath, 'w') as file:
        file.write(', '.join([str(x) for x in lambdas]))
        file.write('\n')
        file.write(', '.join([str(x) for x in spectrum]))
        file.write('\n')

