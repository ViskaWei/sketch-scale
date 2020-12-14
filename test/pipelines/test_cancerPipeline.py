  
import sys
import os
# import numpy as np
# import pandas as pd
import argparse
# import getpass
sys.path.insert(0,"/home/swei20/sketch-scale/")
os.chdir("/home/swei20/sketch-scale/")
os.getcwd()
from src.pipelines.cancerPipeline import CancerPipeline


def main():
    sys.argv = ['test_cancerPipeline', '--config', "./src/configs/cancerConfig.json"]
    p=CancerPipeline()
    print("==============================================================")
    print(p.args)
    p.execute()

main()