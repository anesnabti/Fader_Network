import tensorflow as tf
import numpy as np
import cv2
import os 
import sys
import pathlib
from fader_networks import GAN
from model import encoder, decoder, discriminator
sys.path.append(str(pathlib.Path().parent.resolve()) + "\\cfg")
from config import debug, info, warning, log_config, PATH, epochs, batch_size
import glob
Nbr_images = 202599
log_config('train')
image_path = glob.glob(PATH + '\\data\\train' + '\\*.jpg')

from fader_networks import GAN


class Train:

    def __init__ (self, num_attr, valeur, lr):
 
        self.image_path = glob.glob(PATH + '\\data\\train' + '\\*.jpg')
        self.gan = GAN(num_attr, valeur)
        self.lr = lr
        self.attr = np.load(PATH + "\\data\\train\\train_images_att.npy")
        self.gan = GAN(encoder=encoder, decoder=decoder, discriminator=discriminateur)

        
    
    def normalize (self, image):  
        return image/127.5-1.0             # return image normalized between -1 and 1 


    def image_batch(self, bacth_size, itr):
        """ provide batch images to train """ 
        tmp = []
        for i in range (bacth_size):
            tmp.append(self.normalize(cv2.imread(np.array(self.image_path)[i + itr * bacth_size])))
        info("Batch size image OK")
        return tmp


    def training(self, epochs, batch_size):

        nbr_itr_per_epoch = int(len(np.array(self.image_path))/batch_size)
         
        for epoch in range (epochs):
         
            for i in range (nbr_itr_per_epoch):
                lambda_e = 0.0001 * (epoch*nbr_itr_per_epoch + i)/(nbr_itr_per_epoch*epochs)

                imgs = self.image_batch(batch_size, i)
                atts = self.attr[i*batch_size : (i+1)*batch_size]
                # for img, att in zip(np.array(imgs), atts):
                self.gan.train_step(imgs, atts, lambda_e)



# if __name__ == '__main__':
#     training = Train([6],[1],0.01)  
#     training.training(3,10)  

