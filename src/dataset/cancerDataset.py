import os
import numpy as np
import logging
from scipy.ndimage import gaussian_filter
from concurrent.futures import ThreadPoolExecutor, as_completed

class CancerDataset():
    def __init__(self, fileDir, nImg=None, isTest=False, smooth= None, size = [1004,1344,35]):
        self.filePath=None
        self.nImg=nImg
        self.isTest = isTest
        self.smooth = smooth
        self.ini=0
        self.size = size
        self.ver, self.hor, self.layer = self.size
        self.data = {}
        self.cov=None
        self.eigenVecs=None
        # ===========================  Load data  ================================
        self.get_file_path(fileDir)
        self.load(parallel=True)

    # ===========================  FUNCTIONS  ================================

    def get_file_path(self, fileDir):
        filePath = [os.path.join(fileDir, f) for f in os.listdir(fileDir) if f.endswith('.fw')]
        if (self.nImg is None) or (self.nImg== -1):
            self.filePath=filePath
            self.nImg=len(filePath)
        else:
            self.filePath = filePath[self.ini:self.ini + self.nImg]
        logging.info("  Loading # {} image(s) ".format(len(self.filePath)))
    
    def load_ith_item(self,idx):
        with open(self.filePath[idx],'rb') as f_id:
            img = np.fromfile(f_id, count=np.prod(self.size), dtype = np.uint16)
            img = np.reshape(img, self.size)

            if self.isTest:             
                img = img[-self.ver:,-self.hor:,:]

            if self.smooth is not None: 
                img=gaussian_filter(img,sigma=self.smooth)
            
            img= np.reshape(img, [self.ver*self.hor, self.layer]).astype('float')
            self.data[idx]=img
        return img.T.dot(img)

    def get_img_loader(self):
        if self.isTest:
            self.ver, self.hor = 400, 300
        logging.info("  Loaded dataset with shapes: {} {}".format(self.ver,self.hor))
        if self.smooth is not None:
            logging.info("  Smoothing with sigma:  {}".format(self.smooth))
        # return self.load_ith_item
     
    def load(self, parallel=True):
        self.get_img_loader()
        if parallel:
            with ThreadPoolExecutor() as executor: 
                    futures = []
                    for idx in range(self.nImg):
                        futures.append(executor.submit(self.load_ith_item, idx))
                        # print(f" No.{idx} image is loaded")
                    self.cov = np.zeros([self.layer,self.layer])
                    for future in as_completed(futures):
                        mul = future.result()
                        self.cov += mul
        else:
            raise "non-parallel loading not implemented yet"

    def get_eigenVecs(self, dim):
        # use svd since its commputational faster
        logging.info("=============== PCA: {} ===============".format(dim))
        u,s,v = np.linalg.svd(self.cov)
        assert np.allclose(u, v.T)
        logging.info('Explained Variance Ratio {}'.format(np.round(s/sum(s),3)))
        self.eigenVecs = u[:,:dim]

    def get_pc(self, dim):
        self.get_eigenVecs(dim)
        pc=[self.data[i].dot(self.eigenVecs) for i in range(self.nImg)]
        pc= np.concatenate(pc)
        logging.info(" PCA output Shape:  {}".format(pc.shape))
        return pc 

