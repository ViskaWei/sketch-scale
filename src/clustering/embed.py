import umap
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


class Embed(object):
    def __init__(self, dfHH, dim, nCluster, ratio = None):
        if ratio is not None:  dfHH = dfHH[dfHH['ra']<ratio] 
        self.ratio = ratio 
        self.dfHH = dfHH
        self.dim = dim
        self.ftr = dfHH.columns[:dim]
        self.nCluster = nCluster or 10
        self.kmap = None

    def run(self):
        self.get_umapT()
        self.get_cluster()


    def get_umapT(self):
        umapT = umap.UMAP(n_components=2, min_dist=0.0, n_neighbors=50, random_state=1178)
        matUMAP = umapT.fit_transform(self.dfHH[self.ftr].values)
        self.dfHH['u1'] = matUMAP[:,0]
        self.dfHH['u2'] = matUMAP[:,1]
        self.umapT = umapT

    def get_mapped(self, df, ftr):
        # lb,ub=int(HH_pd['freq'][0]*lbr),int(HH_pd['freq'][0])
        # HH_pdc=HH_pd[HH_pd['freq']>lb]
        # print(f'lpdc: {len(HH_pdc)} lpd: {len(HH_pd)} ub:{ub} lb:{lb} HHratio:{lbr}')
        matUMAPED=self.umapT.transform(df[ftr])   
        df['u1']=matUMAPED[:,0]
        df['u2']=matUMAPED[:,1]
        return df

    def get_cluster(self, u1 = 'u1', u2 = 'u2'):
        umap_result = self.dfHH.loc[:, [u1, u2]].values
        kmap = KMeans(n_clusters=self.nCluster, n_init=30, algorithm='elkan',random_state=926)
        kmap.fit(umap_result, sample_weight = None)
        self.dfHH[f'k{self.nCluster}'] = kmap.labels_ + 1 
        self.kmap = kmap

    


# def get_freq_aug(df):
#     freq=df['freq'].values
#     df['FN']=freq/freq[-1]
#     df['FR']=df['FN'].apply(lambda x: 1+np.floor(np.log2(x)))
# #     df['RR']=1+np.floor(np.log2(freq/freq[-1]))
#     return freq

# def get_aug_pd(df,ftr,mode='FR'):
#     data=df[ftr].values   
#     freqN =df[mode].values
#     freqInt=freqN.astype('int')
#     plt.figure(figsize=(5,5))
#     _=plt.hist(freqInt,bins=freqInt[0])  
#     data_aug=data[0]
#     freq_list=[]
#     np.random.seed(112)
#     for ii, da in enumerate(data[1:]):
#         freqInt_ii=freqInt[ii]
#         freq_list+=[freqInt_ii]*freqInt_ii
#         randmat=np.random.rand(freqInt_ii,pca_comp)-0.5
#         da_aug = da+0.25*randmat
#         assert np.sum(np.round(da_aug)-da)<0.001
#         data_aug=np.vstack((data_aug,da_aug))
#     data_aug=data_aug[1:]
#     print(data_aug.shape,len(freq_list))
#     aug_pd=pd.DataFrame(data_aug, columns=list(range(pca_comp)))
#     aug_pd['freqInt']=freq_list
#     aug_pd['freq']=aug_pd['freqInt'].apply(lambda x: float(x))    
#     return aug_pd