import h5py
import numpy as np
import pandas as pd
import pickle
import logging


def save_dataset(saveDir,data,dataName, name=None, fileFormat=None, suffix=None):
    if fileFormat=="h5":
        with h5py.File(f'{saveDir}/{name}.hdf5', 'w') as f:
            f.create_dataset(dataName, shape=data.shape, data=data)  
    elif fileFormat == "pickle":
        if suffix is None: suffix = 'txt'
        pickle.dump(data, open(f'{saveDir}/{dataName}.{suffix}','wb'))
    elif fileFormat == "csv":
        data.to_csv(f'{saveDir}/{dataName}.csv', index=False)
    elif fileFormat == "txt":
        np.savetxt(f'{saveDir}/{dataName}.txt', data)
    else:
        raise "error saving"


def load_dataset(saveDir, dataName, name=None, fileFormat=None, suffix=None):
    if fileFormat=="h5":
        with h5py.File(f'{saveDir}/{name}.hdf5', 'r') as f: 
            data = f[dataName][()]
        logging.info(f"loading {dataName} of size {data.shape}")
    elif fileFormat =="pickle":
        data = pickle.load(open(f'{saveDir}/{dataName}.{suffix}','rb')) 
    elif fileFormat == "csv":
        data = pd.read_csv(f'{saveDir}/{dataName}.csv')
    elif fileFormat == "txt":
        data = np.loadtxt(f'{saveDir}/{dataName}.txt')
    else:
        raise "error loading"
    return data
