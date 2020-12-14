import logging
import os
import time
import copy
import math
import json
import random
import numpy as np
from src.pipelines.basePipeline import BasePipeline
from src.dataset.save import load_dataset
from src.sketch.SnS import SnS
from src.clustering.embed import Embed


class SketchPipeline(BasePipeline):
    def __init__(self, dfNorm=None, logging=True):
        super().__init__()
        self.sketchMode='exact'
        self.dfNorm= dfNorm
        self.base=None
        self.sketchMode = None
        self.dtype = "uint64"
        self.topk = 20000
        self.csParams = None
        self.isSave = {'stream':False, 'HH':False, "UMAP": False, "KMAP": False}
        self.ratio = None
        self.nCluster = None
        self.dfHH = None
        self.dfEmbed = None

      
        
    def add_args(self, parser):
        super().add_args(parser)
        ############################## ENCODE #######################################
        parser.add_argument('--base', type=int, default=None, help='Base\n')
        parser.add_argument('--dtype', type=str, default=None, help='dtype\n')
        parser.add_argument('--dfNorm', type=str, default=None, help='dtype\n')
        parser.add_argument('--saveStream', type=bool, help='Saving stream\n')
        
        ############################## SKETCH #######################################
        parser.add_argument('--sketchMode', type=str, help='exact or cs\n')
        parser.add_argument('--topk', type=int, help='keep top k Heavy Hitters\n')
        parser.add_argument('--csParams', type=str, nargs=4, help='count sketch table parameters\n')


        parser.add_argument('--ratio', type=int, default=None, help='ratio of HHs\n')
        parser.add_argument('--saveHH', type=bool, help='Saving HH\n')
        
        ############################## EMBED #######################################
        parser.add_argument('--nCluster', type=int, default=None, help='KMEAN cluster numbers\n')
        parser.add_argument('--saveUMAP', type=bool, help='Saving UMAPT\n')
        parser.add_argument('--saveKMAP', type=bool, help='Saving KMAP\n')



    def prepare(self):
        super().prepare()
        self.apply_model_args()


    def apply_model_args(self):
        self.apply_encode_args()
        self.apply_sketch_args()
        self.apply_cluster_args()
        self.apply_save_args()

            
    def apply_encode_args(self):
        if self.dfNorm is None:
            if 'dfNorm' in self.args and self.args['dfNorm'] is not None:
                self.dfNorm = load_dataset(None, None, fileFormat="csv", dirPath = self.args['dfNorm'])
        if 'base' in self.args and self.args['base'] is not None:
            self.base=self.args['base']
        else:
            raise "--base base not specified"
        if 'dtype' in self.args and self.args['dtype'] is not None:
            self.dtype=self.args['dtype']
        if 'ratio' in self.args and self.args['ratio'] is not None:
            self.ratio=self.args['ratio']

    def apply_sketch_args(self):
        if 'sketchMode' in self.args and self.args['sketchMode'] is not None:
            self.sketchMode=self.args['sketchMode']
        if 'topk' in self.args and self.args['topk'] is not None:
            self.topk=self.args['topk']
            if self.isTest: 
                self.topk = 1000
                logging.info(f"Taking {self.topk} only for testing")

    def apply_cluster_args(self):
        if 'nCluster' in self.args and self.args['nCluster'] is not None:
            self.nCluster=self.args['nCluster']
        
    def apply_save_args(self):
        if 'saveStream' in self.args and self.args['saveStream'] is not None:
            self.isSave['stream']=self.args['saveStream']
        if 'saveHH' in self.args and self.args['saveHH'] is not None:
            self.isSave['HH']=self.args['saveHH']
        if 'saveUMAP' in self.args and self.args['saveUMAP'] is not None:
            self.isSave['UMAP']=self.args['saveUMAP']
        if 'saveKMAP' in self.args and self.args['saveKMAP'] is not None:
            self.isSave['KMAP']=self.args['saveKMAP']
        logging.info('saving {}'.format(self.isSave.items()))


    def sketch(self):
        self.dfHH = self.run_step_SnS()
        self.dfEmbed = self.run_step_cluster(self.dfHH) 

    def run_step_SnS(self):
        sns=SnS(self.dfNorm, self.base, sketchMode = self.sketchMode,\
                 topk = self.topk, csParams=self.csParams, dtype=self.dtype)
        sns.run()
        if self.isSave['stream']: self.save_dataset(sns.stream, "stream", "h5") 
        print(sns.dfHH)
        print(self.isSave)
        if self.isSave['HH']: self.save_dataset(sns.dfHH, "dfHH", "csv") 
        return sns.dfHH
        
    def run_step_cluster(self, dfHH):
        embed = Embed(dfHH, ratio = self.ratio, inDim = self.dim, nCluster = self.nCluster)
        embed.run()
        self.save_dataset(embed.dfHH, "dfEmbed", "csv") 
        if self.isSave['UMAP']: self.save_dataset(embed.umapT, "umapT", "joblib")
        if self.isSave['KMAP']: self.save_dataset(embed.kmap, "kmap", "joblib") 
        return embed.dfHH


    # def run_step_encode(self, dfNorm):
    #     stream=get_encode_stream(dfNorm, self.base, self.dtype)
    #     if self.save['stream']: 
    #         self.save_txt(stream, 'stream')
    #     elif self.idx is not None:
    #         self.save_txt(stream[self.idx[0]:self.idx[1]],f'stream{self.idx[-1]}')
    #     return stream
    
    # def run_step_sketch(self, stream):
    #     if self.sketchMode=='exact':
    #         dfHH=get_dfHH(stream,self.base,self.dim, self.dtype, True, None)
    #     else:
    #         raise 'exact only now'
    #         # dfHH=get_dfHH(stream,base,ftr_len, dtype, False, topk, r=16, d=1000000,c=None,device=None)
    #     if self.save['HH']:   
    #         dfHH.to_csv(f'{self.out}/dfHH_b{self.base}_{self.sketchMode}.csv',index=False)
    #     return dfHH
