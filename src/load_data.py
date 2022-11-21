import tensorflow as tf
import numpy as np
import cv2
import os 
import sys
import pathlib
sys.path.append(str(pathlib.Path().parent.resolve()) + "\\cfg")
from config import debug, info, warning, log_config, PATH
Nbr_images = 202599

""" 
This file will load the pre-processed data.
Split them on train, validation and test.
Normalize images
"""
attributes = np.load(str(pathlib.Path().parent.resolve()) + "\\data\\ATTRIBUTS.npy")
log_config("load_data")


class LoadData : 

    def __init__ (self, train_size, validation_size, test_size):
        if train_size < 0 or validation_size < 0 or test_size < 0 : 
            debug ("Split ratio dataset must be > 0 ")
            raise Exception ("Split ratio dataset must be > 0 ")
        if train_size + validation_size + test_size > 1:
            debug("Sum of split ratio must be equal to 1")
            raise Exception ("Sum of split ratio must be equal to 1")
        if train_size + validation_size + test_size != 1 :
            warning ("Sum of split ratio must be equal to 1")
        if validation_size == 0: 
            warning("Ratio validation is 0")

        self.train_size, self.validation_size, self.test_size = train_size, validation_size, test_size 



    def split_data (self):
        train_path = PATH + '\\data\\train'
        test_path = PATH + "\\data\\test"
        validation_path = PATH + "\\data\\validation"
        
        if not os.path.exists(train_path):
            os.makedirs(train_path)
        if not os.path.exists(test_path):
            os.makedirs(test_path)
        if not os.path.exists(validation_path):
            os.makedirs(validation_path)   
        
        ############# Cpying image train in the folder ...//data//train
        info("start copying train images in ...//data//train")
        for i in range (1, int(Nbr_images*self.train_size) + 1):
            cv2.imwrite(train_path + "\\%06i.jpg" % i, cv2.imread(PATH + "\\data\\img_align_celeba_resize\\%06i.jpg" % i))
        info("Copying train images OK")
        
        ############## Copying test images into ..//data//test
        info("start copying test images in ...//data//test")
        for i in range (int(Nbr_images*self.train_size) + 1, int(Nbr_images*self.train_size) + int(Nbr_images*self.test_size) + 1):
            cv2.imwrite(test_path + "\\%06i.jpg" % i, (cv2.imread(PATH + "\\data\\img_align_celeba_resize\\%06i.jpg" % i)))
        info("Copying test images OK")

        ############## Copying validation images in the folder ..//data//validation
        info("start copying validation images in ...//data//test")
        for i in range (int(Nbr_images*self.train_size) + int(Nbr_images*self.test_size) + 1, int(Nbr_images*self.train_size) + int(Nbr_images*self.test_size) + int(Nbr_images*self.validation_size) + 1):
            cv2.imwrite(validation_path + "\\%06i.jpg" % i, (cv2.imread(PATH + "\\data\\img_align_celeba_resize\\%06i.jpg" % i)))
        info("Copying validation images OK")

        #nbr, nbr_att = len(images), len(attributes)
 
        # train_attributes = attributes[:int(self.train_size * nbr_att)]
        # validation_attributes = attributes[int(self.train_size * nbr_att) : int(self.validation_size * nbr_att)]
        # test_attributes = attributes[int(self.validation_size)* nbr_att : int(self.test_size) * nbr_att]
        # info("Split attributes OK")
        
        

if __name__ == '__main__':
    ld = LoadData(0.7, 0.15, 0.15)
    ld.split_data()
