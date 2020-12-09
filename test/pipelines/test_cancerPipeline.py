  
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

from src.pipelines.cancerPipeline import CancerPipeline


def main():
    sys.argv = ['test_cancerPipeline', '--config', "/home/swei20/sketch-scale/src/configs/cellConfig_test.json"]
    p=CancerPipeline()
    p.prepare()
    p.run()
# isTest=True
# isSmooth=False
# dataDir = r'./data/bki'  

main()