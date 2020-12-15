import logging

from src.pipelines.sketchPipeline import SketchPipeline
from src.dataset.cancerDataset import CancerDataset
from src.prepro.cancerPrepro import CancerPrepro
from src.dataset.save import save_dataset, load_dataset


class CancerPipeline(SketchPipeline):
    def __init__(self, logging=True):
        super().__init__()
        self.nImg=None
        self.smooth=None
        self.cutoff=None
        self.addSave={'mat': False, 'mask':False, 'maskId':None}
        self.idx=None
        
    def add_args(self, parser):
        super().add_args(parser)
        # ===========================  LOAD  ================================
        parser.add_argument('--nImg', type=int, help='num of image loading\n')
        parser.add_argument('--smooth', type=float, default=None, help='Gaussian smooth sigma\n')
        parser.add_argument('--saveMat', type=bool, help='Saving mat\n')
        parser.add_argument('--saveMask', type=bool, help='Saving mask\n')
        parser.add_argument('--maskId', type=int, help='Id of mask saved\n')
        # ===========================  PREPRO  ================================
        parser.add_argument('--cutoff', type=str, default=None, help='Bg cutoff\n')

    # ===========================  PREPARE  ================================
    
    def prepare(self):
        super().prepare()
        self.apply_dataset_args()
        self.apply_prepro_args()
        self.apply_cancer_save_args()

    def apply_dataset_args(self):
        if 'nImg' in self.args and self.args['nImg'] is not None:
            self.nImg=self.args['nImg']
            if self.isTest: self.nImg = 1
        
        if 'smooth' in self.args and self.args['smooth'] is not None:
            self.smooth=[self.args['smooth'],self.args['smooth'],0]

    def apply_prepro_args(self):
        if 'cutoff' in self.args and self.args['cutoff'] is not None:
            try:
                self.cutoff=load_dataset(self.args['cutoff'], fileFormat="pickle") 
            except:
                self.cutoff=None 
                logging.info('cannot load cutoff, calculating again')
        elif self.isTest: 
            self.cutoff = 0
    
    def apply_cancer_save_args(self):
        if 'saveMat' in self.args and self.args['saveMat'] is not None:
            self.addSave['mat']=self.args['saveMat']
        if 'saveMask' in self.args and self.args['saveMask'] is not None:
            self.addSave['mask']=self.args['saveMask']
        if self.addSave['mask']:
            if 'maskId' in self.args and self.args['maskId'] is not None:
                self.addSave['maskId'] = self.args['maskId']
        logging.info('saving {}'.format(self.addSave.items()))
        
    # ===========================  RUN  ================================

    def run(self):
        super().run()
        matPCA = self.run_step_load()
        self.run_step_prepro(matPCA)
        self.run_step_embed()
        self.cluster()


    def run_step_load(self):
        ds=CancerDataset(self.inDir ,nImg=self.nImg, isTest=self.isTest, smooth=self.smooth)
        self.nImg=ds.nImg
        if self.addSave['maskId'] is not None:
            if self.addSave['maskId'] >self.nImg:
                self.addSave['maskId'] = 0
                logging.info('maskId out of range, saving 0th img')
        matPCA = ds.get_pc(self.dim)
        del ds
        if self.addSave['mat']: self.save_dataset(matPCA, 'matPCA', "txt")        
        return matPCA  

    def run_step_prepro(self, matPCA):
        prepro=CancerPrepro(matPCA, cutoff=self.cutoff)
        assert prepro.dim == self.dim
        self.dfNorm, mask = prepro.get_cancer_norm()
        if self.cutoff is None: self.save_dataset(prepro.cutoff, "cutoff" , "pickle")
        logging.info(" cutoff @:  {}".format(prepro.cutoff))
        del prepro
        if self.addSave['mask']: self.save_mask(mask,'mask')

    def run_step_save(self, dfNorm):
        self.save_dataset( dfNorm, "dfNorm", "csv")
    
    def save_mask(self,mask, filename):
        mask2d=mask.reshape((self.nImg, 1004*1344))
        if self.addSave['maskId'] is None:
            self.save_dataset(mask, f"{filename}_all", "txt")
        else:
            maskId=self.addSave['maskId']
            mask0= mask2d[maskId]
            idxi=int(mask2d[:maskId].sum())
            idxj=int(mask2d[:(maskId+1)].sum())
            assert idxj-idxi == mask0.sum()
            self.idx=[idxi,idxj,maskId]
            logging.info('idxi, idxj, maskId: {}'.format(self.idx))
            logging.info('  saving mask {}{}'.format(mask0.shape, mask.sum()))
            self.save_dataset(mask0, f"{filename}Id{maskId}", "txt")

    # ===========================  EVALUATION  ================================
    
    def eval(self):
        super().eval()
        pass




    
   

 