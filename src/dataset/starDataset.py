import os
import numpy as np
import pandas as pd

import logging
from src.prepro.normFn import get_min_max_norm



class StarDataset():
    def __init__(self, fileDir, photoName, specName, ftr, isTest=False):
        self.fileDir=None
        self.isTest = isTest
        self.photoPath = None
        self.specPath = None
        self.ftr = ['ug', 'ur', 'ui', 'uz', 'gz', 'gi', 'gr', 'rz', 'ri', 'iz']

        self.dim = len(ftr)
    # ===========================  LOADING  ================================
        self.get_file_path(photoName, specName)





    # ===========================  FUNCTIONS  ================================

    def get_file_path(self, photoName, specName):
        self.photoPath = self.fileDir + photoName
        self.specPath = self.fileDir + specName
        if self.isTest:
            self.photoPath += '_test'
            self.specPath += 'test'

    def get_photo_norm(self):
        dfPhoto=pd.read_csv(self.photoPath)
        dfPhoto = dfPhoto[self.ftr]


    def prepro_photo_spec(self):
        dffull=pd.read_csv(self.photoPath)
        dffull=dffull[dffull['class']!='GALAXY']
        print(dffull.shape)
        dfspec_norm,vmin,vrng,vmean,vstd,df_lbl=get_std_norm_pd(dffull[ftr],dffull[['class','subclass','lu']])
        # print(vmin,vrng,vmean,vstd)
        # print(dfspec_norm.all().isnull().sum())
        df_lbl=get_stellar_pd(df_lbl)  
        # dfspec_norm.to_csv(f'{wpath}/spec_norm.csv',index=False)
        print(dfspec_norm.shape)
        # df_lbl.to_csv(f'{wpath}/spec_lbl.csv', index=False)   
        dfphoto=pd.read_csv(PHOTO_DATA)
        dfphoto_norm=(dfphoto[ftr]-vmean)/vstd
        dfphoto_norm=(dfphoto_norm-vmin)/vrng
        dfphoto_norm=dfphoto_norm[(dfphoto_norm[dfphoto_norm.columns] >= 0).all(axis=1) & (dfphoto_norm[dfphoto_norm.columns] <= 1).all(axis=1)]
        assert (dfphoto_norm.min().min()>=0) & (dfphoto_norm.max().max()<=1)
        print(dfphoto_norm.shape)
            # df.to_csv(f'{wpath}/photo_norm_{base}.csv', index=False)
        return dfphoto_norm, dfspec_norm,df_lbl
    
    
    def prepro_std_specs(SPEC_DATA, ftr=None, sig=3.0, w=True,wpath=None):
        df_full=pd.read_csv(SPEC_DATA)
        df_snorm,vmean,vstd,df_label=get_prepro_std_pds(df_full,ftr=ftr, lbl_str=['class','subclass'],sig=sig)
        print(df_snorm.head(),vmean,vstd)
        print('label',df_label.head() )
        if w:
            print(f'writing to {wpath}')
            np.savetxt(f'{wpath}/vmean.txt',vmean)
            np.savetxt(f'{wpath}/vstd.txt',vstd)
            df_snorm.to_csv(f'{wpath}/spec_norm.csv',index=False)
            df_label.to_csv(f'{wpath}/spec_lbl.csv', index=False)
        return df_snorm,vmean,vstd,df_label



    def get_file_path(self, fileDir):
        filePath = [os.path.join(fileDir, f) for f in os.listdir(fileDir) if f.endswith('.fw')]
        if (self.nImg is None) or (self.nImg== -1):
            self.filePath=filePath
            self.nImg=len(filePath)
        else:
            self.filePath = filePath[self.ini:self.ini + self.nImg]
        logging.info("  Loading # {} image(s) ".format(len(self.filePath)))
        
