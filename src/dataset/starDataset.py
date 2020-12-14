import os
import numpy as np
import pandas as pd

import logging
from src.prepro.normFn import get_min_max_norm




class StarDataset():
    def __init__(self, rootDir, isTest=True, ftr=None, starDir=None, photoName=None, specName=None):
        self.rootDir=rootDir
        self.starDir = starDir or "/DR13color/"
        self.photoName = photoName or "photo30M_ext"
        self.specName = specName or "spec530k"
        self.ftr = ftr or  ['ug', 'ur', 'ui', 'uz', 'gz', 'gi', 'gr', 'rz', 'ri', 'iz']
        self.isTest = isTest
        self.photoPath = None
        self.specPath = None
        self.lblPath = None
        self.vmin = None
        self.vmax = None
        self.dim = len(self.ftr)
        self.dfPhotoNorm = None
        self.dfSpecNorm = None
        self.dfLabel = None

    # ===========================  LOADING  ================================

        self.get_file_path()

    # ===========================  FUNCTIONS  ================================

    def run(self):
        self.get_photo_norm()
        self.get_spec_norm()


    def get_file_path(self):
        if self.isTest:
            self.photoName += '_test'
            self.specName += '_test'
        self.photoPath = self.rootDir + self.starDir + self.photoName + ".csv"
        self.specPath = self.rootDir + self.starDir + self.specName + ".csv"
        self.lblPath = self.rootDir + "label.csv"


    def get_photo_norm(self):
        dfPhoto=pd.read_csv(self.photoPath)
        dfPhoto = dfPhoto[self.ftr]
        self.dfPhotoNorm, self.vmin, self.vmax = get_min_max_norm(dfPhoto, out=True)
                
    def get_spec_norm(self, labelName = ['class','subclass']):
        df = pd.read_csv(self.specPath)
        df = df[(df['class']=='STAR' )|(df['class']=='QSO')]    
        self.dfLabel = self.get_subclass_lbl(df[labelName])
        self.dfSpecNorm = (df[self.ftr] - self.vmin)/(self.vmax - self.vmin)
        

    def get_subclass_lbl(self, dfLabel):
        dfLabelDict=pd.read_csv(self.lblPath)
        dictLu={}
        dictLbl={}
        dictMain={}
        for ii, subcls in enumerate(dfLabelDict['subclass'].values):
            dictLbl[subcls]=dfLabelDict['T'][ii]
            dictLu[subcls]=dfLabelDict['L'][ii]
            dictMain[subcls]=dfLabelDict['class5'][ii]   
        dfLabel['subclass'][dfLabel['subclass'].isnull()]='qN'                
        dfLabel['subclass'][dfLabel['subclass']=='']='qN'
        dfLabel['lbl']=dfLabel['subclass'].apply(lambda x: dictLbl[x])
        dfLabel['lu']=dfLabel['subclass'].apply(lambda x: dictLu[x].strip())
        dfLabel['class5']=dfLabel['subclass'].apply(lambda x: dictMain[x])
        print(dfLabel['class5'].unique())
        print(dfLabel['lu'].unique())
        dfLabel['t']=dfLabel['lbl'].apply(lambda x: str(x)[0])
        dfLabel['t8']=dfLabel['t']
        dfLabel['t8'][(dfLabel['t']=='L')|(dfLabel['t']=='T')]='M'
        dfLabel['t8'][(dfLabel['t']=='O')|(dfLabel['t']=='B')]='A'
        return dfLabel

