import tensorflow as tf
import numpy as np
import cv2
import os 
import sys
import pathlib
from load_data import Loader
from fader_networks import GAN
from model import Encoder, Decoder, Discriminator
sys.path.append( os.path.dirname(os.path.abspath(__file__))[:-3] + "\\cfg")
from config import debug, info, warning, log_config, PATH
import glob
Nbr_images = 202599
log_config('train')
image_path = glob.glob(PATH + '\\data\\train' + '\\*.jpg')
from fader_networks import GAN

class Train:

    def __init__ (self, lr):
 
        self.image_path = glob.glob(PATH + '\\data\\train' + '\\*.jpg')
        # self.gan = GAN(num_attr, valeur)
        self.lr = lr
        self.gan = GAN(encoder=Encoder(), decoder=Decoder(disc = True), discriminator=Discriminator())

        

    def training(self, epochs, batch_size):
        ld = Loader()
        nbr_itr_per_epoch = int(len(self.image_path)/batch_size)
        info('start training') 
        for epoch in range (epochs):
         
            for i in range (nbr_itr_per_epoch):
                lambda_e = 0.0001 * (epoch*nbr_itr_per_epoch + i)/(nbr_itr_per_epoch*epochs)

                imgs, atts = ld.Load_Data(batch_size,i)
                #for img, att in zip(np.array(imgs), atts):
                loss_model = self.gan.train_step(imgs, atts, lambda_e)

            print(f'epoch : {epoch}  --------   loss = {loss_model}')

        info(f"epoch: {epoch} finished OK")

# if __name__ == '__main__':
    # training = Train(0.01)

    # training.training(epochs=1, batch_size=32)  



