import copy
import time
import torch
import numpy as np
import pandas as pd
from collections import Counter
from src.sketch.csvec import CSVec

class SnS(object):
    def __init__(self, dfNorm, base, dtype=None, sketchMode=None,\
                             topk = 20000, csParams=None):
        self.sketchMode=sketchMode or "exact"
        self.dfNorm= dfNorm
        self.base = base
        self.dtype = dtype or "uint64"
        self.dim = len(dfNorm.columns)
        self.stream = None
        self.topk = topk
        self.csParams = csParams   # d, r, c, device
        self.HH = None
        self.freq = None
        self.dfHH = None

    def run(self):
        self.get_encode_stream()
        self.get_HH_pd()
        
######################################## ENCODE ########################################
    def get_encode_stream(self):
        mat=(self.dfNorm*(self.base-1)).round()
        assert (mat.min().min()>=0) & (mat.max().max()<=self.base-1)
        streamEncode=self.horner_encode(mat) 
        matDecode=self.horner_decode(streamEncode)  
        assert (matDecode.min().min()>=0) & (matDecode.max().max()<=self.base-1)
        try:
            assert np.sum(abs(matDecode-mat.values))<=1e-4   
        except:
            print(np.nonzero(np.sum(abs(matDecode-mat.values),axis=1)), np.sum(abs(matDecode-mat.values)))
            raise 'overflow, try lower base or fewer features'     
        self.stream = streamEncode

    def horner_encode(self, mat):
        r,c=mat.shape
        print('samples:',r,'ftrs:',c, 'base:',self.base)
        streamEncode=np.zeros((r),dtype=self.dtype)
        for ii, key in enumerate(mat.keys()):
            val=(mat[key].values).astype(self.dtype)
            streamEncode= streamEncode + val*(self.base**ii)
    #         print(ii,val, encode)
        return streamEncode

    def horner_decode(self, streamEncode):
        arr=copy.deepcopy(np.array(streamEncode))
        matDecode=np.zeros((len(arr),self.dim), dtype=self.dtype)
        for ii in range(self.dim-1,-1,-1):
            digits=arr//(self.base**ii)
            matDecode[:,ii]=digits
            arr= arr% (self.base**ii)
    #         print(digits,arr)
        return matDecode

######################################## HeavyHitter ########################################
    def get_HH_pd(self):
        if self.sketchMode == "exact":
            self.get_exact_HH()
        else:
            self.get_CS_HH()
        matDecode=self.horner_decode(self.HH)
        assert (matDecode.min().min()>=0) & (matDecode.max().max()<=self.base-1)
        dfHH=pd.DataFrame(matDecode, columns=list(range(self.dim))) 
        dfHH['HH'] = self.HH
        dfHH['freq'] = self.freq
        dfHH['rk']=dfHH['freq'].cumsum()
        dfHH['ra']=dfHH['rk']/dfHH['rk'].values[-1]
        self.dfHH = dfHH

    def get_exact_HH(self):
        print(f'=============exact counting HHs==============')
        t0=time.time()
        dfHH=np.array(Counter(self.stream).most_common(self.topk))
        t=time.time()-t0
        print('exact counting time:{:.2f}'.format(t))
        self.HH, self.freq = dfHH[:,0], dfHH[:,1]
        
    def get_CS_HH(self):
        d, r, c, device = self.csParams
        if c is None: c=10*self.topk 
        stream_tr=torch.tensor(self.stream, dtype=torch.int64)
        csv = CSVec( d, r, c, self.topk, device=device)
        t0=time.time()
        for ii in range(stream_tr.shape[0]//d+1):
            substream=stream_tr[ii*d:(ii+1)*d]
            csv.accumulateVec(substream)
        HHs=stream_tr.unique()
        tfreqs=csv.query(HHs).cpu().numpy()
        tfreqs=np.clip(tfreqs,0,None)
        idx=np.argsort(-1*tfreqs)    
        HHfreq=np.vstack((HHs.numpy(),tfreqs))
        t=time.time()-t0
        print('sketch counting time:{:.2f}'.format(t))
        self.HH, self.freq = HHfreq[:,idx][:,:self.topk]

###############################################################################
