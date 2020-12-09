import h5py
import numpy as np
import pickle




def save_dataset(saveDir,data,dataName, name=None, fileFormat=None):
    if fileFormat=="h5":
        with h5py.File(f'{saveDir}/{name}.hdf5', 'w') as f:
            f.create_dataset(dataName, shape=data.shape, data=data)  
    elif fileFormat == "pickle":
        pickle.dump(data, open(f'{saveDir}/{dataName}.txt','wb'))
    else:
        raise "error saving"


def load_dataset(saveDir, dataName, name=None, sl=None, fileFormat=None):
    if fileFormat=="h5":
        with h5py.File(f'{saveDir}/{name}.hdf5', 'r') as f: 
            data = f[dataName][sl]
        print(f"loading {dataName} of size {data.shape}")
    elif fileFormat =="pickle":
        data = pickle.load(open(saveDir,'rb')) 
    else:
        raise "error loading"
    return data