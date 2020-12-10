  
import sys
import os
# import numpy as np
# import pandas as pd
import argparse
# import umap
# import joblib
# import getpass
# print(os.getcwd())
sys.path.insert(0,"/home/swei20/sketch-scale/")

from src.pipelines.sketchPipeline import SketchPipeline


def main():
    sys.argv = ['test_sketchPipeline', '--config', "./configs/sketchConfig.json"]
    p=SketchPipeline()
    p.prepare()
    p.run()
# isTest=True
# isSmooth=False
# dataDir = r'./data/bki'  

main()