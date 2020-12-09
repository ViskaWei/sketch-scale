import pandas as pd
import umap
import numpy as np
import copy
# from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

import joblib

model_dict={}
# def get_mapping_pd(ftr_pd,umapT,ftr):
#     try: df_umap=ftr_pd[ftr]
#     except: df_umap=ftr_pd[list(map(str,ftr))]
#     u_da=umapT.transform(df_umap.values)
#     # umapped=pd.DataFrame(u_da, columns=[f"u{ii}" for ii in range(len(u_da[0]))])
#     # ftr_pd=pd.concat([ftr_pd,umapped],axis=1)
#     for ii in range(len(u_da[0])):
#         ftr_pd[f'u{ii}'] = u_da[:,ii]
#     print(ftr_pd.keys())
#     return ftr_pd

def get_umap_pd(dfHH, ftr):
    try: df_umap=dfHH[ftr]
    except: df_umap=dfHH[list(map(str,ftr))]
    umapT = umap.UMAP(n_components=2,min_dist=0.0,n_neighbors=50, random_state=1178)
    umap_result = umapT.fit_transform(df_umap.values)
    dfHH['u1'] = umap_result[:,0]
    dfHH['u2'] = umap_result[:,1]
    return umapT

def get_mapping_pd(dfHH,umapT,ftr_len):
    # lb,ub=int(HH_pd['freq'][0]*lbr),int(HH_pd['freq'][0])
    # HH_pdc=HH_pd[HH_pd['freq']>lb]
    # print(f'lpdc: {len(HH_pdc)} lpd: {len(HH_pd)} ub:{ub} lb:{lb} HHratio:{lbr}')
    u_da=umapT.transform(dfHH[list(range(ftr_len))])   
    dfHH['u1']=u_da[:,0]
    dfHH['u2']=u_da[:,1]
    # sns.scatterplot('u1','u2',data=HH_pdc,alpha=0.7,s=10, color='k', marker="+")
    return dfHH

def get_kmean_lbl(dfHH, N_cluster, u1 = 'u1', u2 = 'u2'):
    umap_result = dfHH.loc[:,[u1, u2 ]].values
    kmap = KMeans(n_clusters=N_cluster,n_init=30, algorithm='elkan',random_state=1178)
    kmap.fit(umap_result, sample_weight = None)
    dfHH[f'k{N_cluster}'] = kmap.labels_ + 1 
    return kmap


def get_pred_stream(stream_1D, mask, pd,  k_cluster, val = 'val', bg = -1, color = 0, sgn = 1 ):
    mask=mask.astype('bool')
    HHvals = np.array(pd[val])
    HHcluster = np.array(pd[k_cluster])
    masked_streams = np.zeros(np.shape(stream_1D))
    for idx, val in enumerate( HHvals): 
        label = HHcluster[idx]
        masked_streams = np.where(stream_1D != val, masked_streams, color +sgn * label)
    final_umap = np.ones(mask.shape) * bg
    final_umap[mask] = masked_streams
    return final_umap 

def get_freq_aug(df):
    freq=df['freq'].values
    df['FN']=freq/freq[-1]
    df['FR']=df['FN'].apply(lambda x: 1+np.floor(np.log2(x)))
#     df['RR']=1+np.floor(np.log2(freq/freq[-1]))
    return freq

def get_aug_pd(df,ftr,mode='FR'):
    data=df[ftr].values   
    freqN =df[mode].values
    freqInt=freqN.astype('int')
    plt.figure(figsize=(5,5))
    _=plt.hist(freqInt,bins=freqInt[0])  
    data_aug=data[0]
    freq_list=[]
    np.random.seed(112)
    for ii, da in enumerate(data[1:]):
        freqInt_ii=freqInt[ii]
        freq_list+=[freqInt_ii]*freqInt_ii
        randmat=np.random.rand(freqInt_ii,pca_comp)-0.5
        da_aug = da+0.25*randmat
        assert np.sum(np.round(da_aug)-da)<0.001
        data_aug=np.vstack((data_aug,da_aug))
    data_aug=data_aug[1:]
    print(data_aug.shape,len(freq_list))
    aug_pd=pd.DataFrame(data_aug, columns=list(range(pca_comp)))
    aug_pd['freqInt']=freq_list
    aug_pd['freq']=aug_pd['freqInt'].apply(lambda x: float(x))    
    return aug_pd

