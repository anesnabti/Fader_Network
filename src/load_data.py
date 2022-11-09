import tensorflow as tf
import numpy as np
import cv2
import os 
import sys
import pathlib
sys.path.append(str(pathlib.Path().parent.resolve()) + "\\cfg")
from config import debug, info, warning, log_config

""" 
This file will load the pre-processed data.
Split them on train, validation and test.
Normalize images
"""

images = np.load(str(pathlib.Path().parent.resolve()) + "\\data\\IMAGES.npy")
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
        nbr, nbr_att = len(images), len(attributes) 

        train_images = images[:int(self.train_size * nbr)]
        validation_images = images[int(self.train_size * nbr) : int(self.validation_size * nbr)]
        test_images = images[int(self.validation_size)* nbr : int(self.test_size) * nbr]
        info("Split images OK")
        train_attributes = attributes[:int(self.train_size * nbr_att)]
        validation_attributes = attributes[int(self.train_size * nbr_att) : int(self.validation_size * nbr_att)]
        test_attributes = attributes[int(self.validation_size)* nbr_att : int(self.test_size) * nbr_att]
        info("Split attributes OK")
        return train_images, validation_images, test_images, train_attributes, validation_attributes, test_attributes 
        

    def normalize (self):
        return self.split_data()[0] / 255.0 , self.split_data()[1] / 255.0 , self.split_data()[2] / 255.0 


    def get_data(self):
        return self.normalize(), self.split_data()[3], self.split_data()[4], self.split_data()[5]



# if __name__ == '__main__':
#     ld = LoadData(0.7, 0.15, 0.15)
#     x_train, x_valid, x_test = ld.split_data()

#     print((x_train.size()))