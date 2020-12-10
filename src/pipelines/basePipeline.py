import os
import sys
import json
import logging
import argparse
import numpy as np
from src.dataset.save import save_dataset


class BasePipeline():
    '''
    Setting up pipline for taking command line argment specifying output, name, dim, seed, config
    '''
 
    def __init__(self):
        self.parser = None
        self.args = None
        self.out = None
        self.name='test'
        self.dim=None
        self.debug=False
                
    def prepare(self):
        self.create_parser()
        self.parse_args() 
        self.setup_logging()
        self.apply_args()
        
    def add_args(self,parser):
        parser.add_argument('--config', type=str, help='Load config from json file.')
        parser.add_argument('--seed', type=int, help='Set random\n' )
        parser.add_argument('--name', type=str, help='save model name\n')
        parser.add_argument('--out', type=str, help='output dir\n')
        parser.add_argument('--dim', type=int, default=None,  help='Latent Representation dimension\n')
        

    def create_parser(self):
        self.parser = argparse.ArgumentParser()
        self.add_subparsers(self.parser)

    def add_subparsers(self, parser):
        self.add_args(parser)
    
    def parse_args(self):
        if self.args is None:
            self.args = self.parser.parse_args().__dict__
            self.get_configs(self.args)
        print(self.args)
    
    def is_arg(self, name, args =None):
        args = args or self.args
        return name in args and args[name] is not None

    def get_arg(self, name, args=None, default=None):
        args = args or self.args
        if name in args and args[name] is not None: 
            return args[name]
        elif default is not None:
            return default
        else:
            raise f'arg {name} not specified in command line'
    
    def get_loop_from_arg(self, name, fn = None):
        loopArgs = self.get_arg(name)
        if fn is None:
            loopFn = lambda x: int(loopArgs[0]**(loopArgs[1] + x))
        else:
            loopFn =fn(loopArgs)
        return [loopFn(ii) for ii in range(loopArgs[2])]

    def get_config_args(self, args=None):
        args = args or self.args
        for key, val in args.items():
            if self.is_arg(key,args=args):
                self.args[key] = val
    
    def update_nested_configs(self,args):
        args = self.load_args_jsons(args)
        while self.is_arg('config',args=args): 
            configArg = self.load_args_jsons(args)
            del args['config']
            args.update(configArg)
        return args

    def get_configs(self, args):
        args = self.update_nested_configs(args)
        # print('configArgs', args)
        self.get_config_args(args)
        # print('get_configs_final',self.args)
    
    def load_args_jsons(self, args):
        configFiles = self.get_arg('config', args=args)
        if type(configFiles) is not list:
            configFiles = [configFiles] 
        for configFile in configFiles:
            configArg = self.load_args_json(configFile)
            args.update(configArg)
        return args

    def load_args_json(self, filename):
        # print(filename)
        with open(filename, 'r') as f:
            args = json.load(f)
        return args
    
##################################################APPLY ARGS###############
    def apply_args(self):
        self.apply_init_args()
        self.apply_input_args()
        self.apply_output_args()
    
    def apply_init_args(self):
        if 'seed' in self.args and self.args['seed'] is not None:
            np.random.seed(self.args['seed'])  
        else:
            np.random.seed(112)  
        if 'name' in self.args and self.args['name'] is not None:
            self.name =self.args['name']

    def apply_input_args(self):
        if 'dim' in self.args and self.args['dim'] is not None:
            self.dim = self.args["dim"]
        else:
            raise "--dim latent dimension not specified"
  
    def apply_output_args(self):        
        if 'out' in self.args and self.args['out'] is not None:
            self.out = self.args['out'] 
            self.create_output_dir(self.out, cont=False)
        else:
            raise "--out output directory is not specified"
            
    def create_output_dir(self, dir, cont=False):
        logging.info('Output directory is {}'.format(dir))
        if cont:
            if os.path.exists(dir):
                logging.info('Found output directory.')
            else:
                raise Exception("Output directory doesn't exist, can't continue.")
        elif os.path.exists(dir):
            if len(os.listdir(dir)) != 0:
                print('Output directory not Empty, Replacing might occurs')
        else:
            logging.info('Creating output directory {}'.format(dir))
            os.makedirs(dir)
       

    def init_logging(self, outdir):
        self.setup_logging(os.path.join(outdir, type(self).__name__.lower() + '.log'))
        # handler = logging.StreamHandler(sys.stdout)
        # handler.setLevel(logging.DEBUG)
        # handler.setFormatter(formatter)
        # root.addHandler(handler)
    
    def setup_logging(self, logfile=None):
        root = logging.getLogger()
        root.setLevel(self.get_logging_level())
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    def get_logging_level(self):
        if self.debug: 
            return logging.DEBUG
        else: 
            return logging.INFO

    def save_data(self, data, dataName, fileFormat=None, suffix=None):
        save_dataset(self.out,data,dataName, name=self.name, fileFormat=fileFormat, suffix=suffix)
