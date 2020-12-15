import pandas as pd
import logging

from src.pipelines.sketchPipeline import SketchPipeline
from src.dataset.starDataset import StarDataset
# from src.prepro.starPrepro import StarPrepro

from src.dataset.save import save_dataset, load_dataset


class StarPipeline(SketchPipeline):
    def __init__(self, logging=True):
        super().__init__()
        self.photoName = None
        self.specName = None
        self.star = None
        self.ftr = None
        self.dfNorm = None
        self.dfSpecNorm = None
        self.dfLabel = None


    def add_args(self, parser):
        super().add_args(parser)
        # ===========================  LOAD  ================================
        parser.add_argument('--star', type=int, help='directory of the star data\n')
        parser.add_argument('--photo', type=int, help='filename of the photometric data\n')
        parser.add_argument('--spec', type=int, help='filename of the spectroscopic data\n')

        parser.add_argument('--ftr', type=int, help='features extracted\n')

        
    def prepare(self):
        super().prepare()
        self.apply_dataset_args()

    def apply_dataset_args(self):
        if 'star' in self.args and self.args['star'] is not None:
            self.star=self.args['star']
        
        if 'photo' in self.args and self.args['photo'] is not None:
            self.photoName=self.args['photo']

        if 'spec' in self.args and self.args['spec'] is not None:
            self.specName=self.args['spec']

        if 'ftr' in self.args and self.args['ftr'] is not None:
            self.ftr=eval(self.args['ftr'])

    def run(self):
        self.run_step_prepro()
        self.run_step_embed()
        


    def run_step_prepro(self):
        ds=StarDataset(self.inDir, isTest=self.isTest, ftr=self.ftr, starDir=self.star, \
                        photoName=self.photoName, specName=self.specName)
        ds.run()
        self.dfNorm = ds.dfPhotoNorm
        self.dfSpecNorm = ds.dfSpecNorm
        self.dfLabel = ds.dfLabel

    def run_step_save(self, dfNorm):
        save_dataset(self.out, dfNorm, "dfNorm", name=self.name, fileFormat="csv")
    
    def eval(self):
        self.run_step_transform()

    def run_step_transform(self):
        matUMAPED = self.embed.get_mapped(self.dfSpecNorm, ftr = None)
        self.dfSpec = pd.DataFrame(matUMAPED, columns = list(range(self.embed.emDim)))
        self.dfSpec = pd.concat([self.dfSpec, self.dfLabel], axis=1)
   

 