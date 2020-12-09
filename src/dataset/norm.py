import logging
import numpy as np
import pandas as pd

class Norm():
    def __init__(self, mat, cutoff=None):
        self.mat=mat
        self.dim=len(mat[0])
        self.cutoff=cutoff

    def get_min_max_norm(self, df):
        vmin,vmax=df.min().min(), df.max().max()
        dfNorm=((df-vmin)/(vmax-vmin))
        assert ((dfNorm>=0) & (dfNorm<=1)).all().all()
        return dfNorm

    ###################### operations unique to cancer data ##########################
    def get_cancer_norm(self):
        intensity = self.get_intensity()
        if self.cutoff is None: self.get_cutoff(intensity, nBins = 100, nSigma = 3)
        return self.get_intensity_norm(intensity)
        
    def get_intensity(self):
        intensity = (np.sum(self.mat**2, axis = 1))**0.5
        return intensity
    
    def get_cutoff(self, intensity, nBins = 100, nSigma = 3):
        para = np.log(intensity[intensity > 1])
        (x,y) = np.histogram(para, bins = nBins)
        y = (y[1]-y[0])/2 + y[:-1]
        assert len(x) == len(y)
        x_max =  np.max(x)
        x_half = x_max//2
        mu = y[x == x_max]
        sigma = abs(y[abs(x - x_half).argmin()] -mu)
        cutoff_log = nSigma * sigma + mu
        self.cutoff = np.exp(cutoff_log).round()

    def get_intensity_norm(self, intensity):
        mask = intensity > self.cutoff
        try: 
            m=np.sum(mask)
            assert  m > 1e3
            logging.info('stream length m = {}'.format(m))
        except:
            raise 'stream size too small, lower cutoff or add samples'
        mask=mask.astype('bool')
        intensityCut=intensity[mask]
        dfPCA=pd.DataFrame(self.mat[mask],columns=[f'd{i}' for i in range(self.dim)])
        dfNorm= np.divide(dfPCA, intensityCut[:,None])
        dfNorm=self.get_min_max_norm(dfNorm)
        return dfNorm, mask


    # def get_intensity_norm(self, intensity, cut):
    #     mask = intensity > cut
    #     logging.info('stream length m = {}'.format(np.sum(mask)))
    #     mask=mask.astype('bool')
    #     intensityCut=intensity[mask]
    #     df_pca=pd.DataFrame(self.mat[mask],columns=[f'd{i}' for i in range(self.dim)])
    #     df_uni= np.divide(df_pca, intensityCut[:,None])
    #     dfNorm=get_minmax_pd(df_uni,r=r, vmin=None, vmax=None)
    #     if ONPCA:
    #         df_p2=get_col_norm_pd(df_pca[[1,2]],r=r,w=False,std=False)
    #         dfNorm=pd.concat([df_p2,dfNorm],axis=1)
    #     if ONINT: 
    #         intensityCut=(intensityCut-np.mean(intensityCut))/np.std(intensityCut)
    #         df_inten=pd.DataFrame(intensityCut, columns=['int'])
    #         df_inten=get_col_norm_pd(df_inten,r=r,w=False,std=False)
    #         dfNorm=pd.concat([df_inten,dfNorm],axis=1)
    #     ftr_len=len(dfNorm.keys())
    #     print(dfNorm)
    #     dfNorm=pd.DataFrame(dfNorm.values, columns=list(range(ftr_len)))
    #     return dfNorm, mask, ftr_len