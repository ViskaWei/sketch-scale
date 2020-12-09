import h5py
import numpy as np


def save_dataset(fileName,data,dataName):
    with h5py.File(f'{fileName}.hdf5', 'w') as f:
        f.create_dataset(dataName, shape=data.shape, data=data)  

def load_dataset(fileName, dataName, sl=None):
    with h5py.File(f'{fileName}.hdf5', 'r') as f: 
        data = f[dataName][sl]
    print(f"loading {dataName} of size {data.shape}")
    return data