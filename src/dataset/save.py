import h5py
import numpy as np


def save_dataset(out,logFlux,wave):
    with h5py.File(f'{o}.hdf5', 'w') as f:
    #     f.create_dataset('normFlux', shape=(14, 15,16094),data=normFlux)
        f.create_dataset('logFlux', shape=logFlux.shape, data=logFlux)  


def load_logFlux(DATASET):
    with h5py.File(DATASET, 'r') as f: 
        logFlux = f['logFlux'][()]
        wave = f['wave'][()]
    print(logFlux.shape)
    return logFlux, wave

def load_dataset(DATASET, sl=None):
    if sl is None: sl = np.s_[:, 6:21, 8, 3, 1,:]
    with h5py.File(DATASET, 'r') as f: 
        flux = f['flux'][sl]
        wave = f['wave'][sl[-1]]
    print(flux.shape)
    return flux, wave