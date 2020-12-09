import logging
import os
import time
import copy
import math
import json
import random
import numpy as np
from src.pipelines.basePipeline import BasePipeline



class SketchPipeline(BasePipeline):
    def __init__(self, dfNorm, logging=True):
        super().__init__()
        self.sketchMode='exact'
        self.dfNorm= dfNorm
        self.base=None
        self.save = {'stream':False, 'HHs':False}




        
        
    def add_args(self, parser):
        super().add_args(parser)
        parser.add_argument('--base', type=int, default=None, help='Base\n')
        parser.add_argument('--dtype', type=str, default=None, help='dtype\n')


        parser.add_argument('--sketchMode', type=str, help='exact or cs\n')
        parser.add_argument('--saveStream', type=bool, help='Saving stream\n')
        parser.add_argument('--saveHHs', type=bool, help='Saving HH\n')


    def prepare(self):
        super().prepare()
        self.apply_model_args()

    def apply_model_args(self):
        self.apply_encode_args()
        self.apply_sketch_args()
        self.apply_save_args()

            
    def apply_encode_args(self):
        if 'base' in self.args and self.args['base'] is not None:
            self.base=self.args['base']
        else:
            raise "--base base not specified"
        if 'dtype' in self.args and self.args['dtype'] is not None:
            self.dtype=self.args['dtype']

    def apply_sketch_args(self):
        if 'sketchMode' in self.args and self.args['sketchMode'] is not None:
            self.sketchMode=self.args['sketchMode']
        
    def apply_save_args(self):
        if 'saveStream' in self.args and self.args['saveStream'] is not None:
            self.save['stream']=self.args['saveStream']
        if 'saveHHs' in self.args and self.args['saveHHs'] is not None:
            self.save['HHs']=self.args['saveHHs']
        logging.info('saving {}'.format(self.save.items()))

    def run(self):
        stream=self.run_step_encode(self.dfNorm)
        HHs = self.run_step_sketch(stream)
        return HHs

    def run_step_encode(self, dfNorm):
        stream=get_encode_stream(dfNorm, self.base, self.dtype)
        if self.save['stream']: 
            self.save_txt(stream, 'stream')
        elif self.idx is not None:
            self.save_txt(stream[self.idx[0]:self.idx[1]],f'stream{self.idx[-1]}')
        return stream
    
    def run_step_sketch(self, stream):
        if self.sketchMode=='exact':
            HH_pd=get_HH_pd(stream,self.base,self.dim, self.dtype, True, None)
        else:
            raise 'exact only now'
            # HH_pd=get_HH_pd(stream,base,ftr_len, dtype, False, topk, r=16, d=1000000,c=None,device=None)
        if self.save['HHs']:   
            HH_pd.to_csv(f'{self.out}/HH_pd_b{self.base}_{self.sketchMode}.csv',index=False)
        return HH_pd
