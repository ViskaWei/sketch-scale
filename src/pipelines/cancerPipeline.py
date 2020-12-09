import logging
import os
import time
import numpy as np
import pickle

from src.dataset.cancerDataset import CancerDataset
from src.pipelines.basePipeline import BasePipeline
from src.dataset.norm import Norm
from src.dataset.save import save_dataset, load_dataset


class CancerPipeline(BasePipeline):
    def __init__(self, logging=True):
        super().__init__()
        self.nImg=None
        self.dim=None
        self.smooth=None
        self.cutoff=None
        self.dtype='uint64'
        self.save={'mat': False, 'mask':False, 'maskId':None}
        self.idx=None
        
    def add_args(self, parser):
        super().add_args(parser)
        # ===========================  LOAD  ================================
        parser.add_argument('--nImg', type=int, help='num of image loading\n')
        parser.add_argument('--test', type=bool, help='Test or original size\n')
        parser.add_argument('--smooth', type=float, default=None, help='Gaussian smooth sigma\n')
        parser.add_argument('--saveMat', type=bool, help='Saving mat\n')
        parser.add_argument('--saveMask', type=bool, help='Saving mask\n')
        parser.add_argument('--maskId', type=int, help='Id of mask saved\n')
        # ===========================  PREPRO  ================================
        parser.add_argument('--cutoff', type=str, default=None, help='Bg cutoff\n')

        
    def prepare(self):
        super().prepare()
        self.apply_dataset_args()
        self.apply_prepro_args()
        self.apply_save_args()

    def apply_dataset_args(self):
        if 'in' not in self.args or self.args['in'] is None:
            raise "--in input directory is not specified"

        if 'nImg' in self.args and self.args['nImg'] is not None:
            self.nImg=self.args['nImg']
        
        if 'smooth' in self.args and self.args['smooth'] is not None:
            self.smooth=[self.args['smooth'],self.args['smooth'],0]

    def apply_prepro_args(self):
        if 'cutoff' in self.args and self.args['cutoff'] is not None:
            try:
                self.cutoff=load_dataset(self.args['cutoff'], fileFormat="pickle") 
            except:
                self.cutoff=None
                logging.info('cannot load cutoff, calculating again')
    
    def apply_save_args(self):
        if 'saveMat' in self.args and self.args['saveMat'] is not None:
            self.save['mat']=self.args['saveMat']
        if 'saveMask' in self.args and self.args['saveMask'] is not None:
            self.save['mask']=self.args['saveMask']
        if self.save['mask']:
            if 'maskId' in self.args and self.args['maskId'] is not None:
                self.save['maskId'] = self.args['maskId']
        logging.info('saving {}'.format(self.save.items()))
        

    def run(self, saveNorm=False):
        matPCA = self.run_step_load()
        dfNorm=self.run_step_norm(matPCA)
        if saveNorm: self.run_step_save(dfNorm)
        return dfNorm


    def run_step_load(self):
        ds=CancerDataset(self.args['in'] ,nImg=self.nImg, isTest=self.args['test'], smooth=self.smooth)
        self.nImg=ds.nImg
        if self.save['maskId'] is not None:
            if self.save['maskId'] >self.nImg:
                self.save['maskId'] = 0
                logging.info('maskId out of range, saving 0th img')
        matPCA = ds.get_pc(self.dim)
        del ds
        if self.save['mat']: save_dataset(self.out, matPCA, 'matPCA',fileFormat="txt")        
        return matPCA  

    def run_step_norm(self, matPCA):
        norm=Norm(matPCA, cutoff=self.cutoff)
        assert norm.dim == self.dim
        dfNorm, mask = norm.get_cancer_norm()
        if self.cutoff is None:
            save_dataset(self.out, norm.cutoff, "cutoff" ,fileFormat="pickle")
        logging.info(" cutoff @:  {}".format(norm.cutoff))
        del norm
        if self.save['mask']: self.save_mask(mask,'mask')
        return dfNorm

    def run_step_save(self, dfNorm):
        save_dataset(self.out, dfNorm, "dfNorm", name=self.name, fileFormat="csv")
    
    def save_mask(self,mask, filename):
        mask2d=mask.reshape((self.nImg,1004*1344))
        if self.save['maskId'] is None:
            name=f'{self.out}/{filename}_all.txt' 
            logging.info('  saving {}'.format(name))
            np.savetxt(name, mask)
        else:
            maskId=self.save['maskId']
            mask0= mask2d[maskId]
            idxi=int(mask2d[:maskId].sum())
            idxj=int(mask2d[:(maskId+1)].sum())
            assert idxj-idxi == mask0.sum()
            self.idx=[idxi,idxj,maskId]
            logging.info('idxi, idxj, maskId: {}'.format(self.idx))
            logging.info('  saving mask {}{}'.format(mask0.shape, mask.sum()))
            np.savetxt(f'{self.out}/{filename}Id{maskId}.txt' , mask0)  

    



    
   

 