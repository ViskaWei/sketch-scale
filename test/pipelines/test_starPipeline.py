  
import sys
import os
import argparse
# import getpass
sys.path.insert(0,"/home/swei20/sketch-scale/")
os.chdir("/home/swei20/sketch-scale/")
os.getcwd()
from src.pipelines.starPipeline import StarPipeline


def main():
    sys.argv = ['test_starPipeline', '--config', "./src/configs/starConfig.json"]
    p=StarPipeline()
    print("==============================================================")
    print(p.args)
    p.execute()

main()