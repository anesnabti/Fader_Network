import tensorflow as tf
import numpy as np
import cv2
import os 
import sys
import pathlib
from datetime import datetime as date
from load_data import Loader
from fader_networks import GAN
from model import Encoder, Decoder, Discriminator
sys.path.append( os.path.dirname(os.path.abspath(__file__))[:-3] + "\\cfg")
from config import debug, info, warning, log_config, PATH, save_loss
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

        

    # def training(self, epochs, batch_size, attributs, weights=""):
    #     name_attributs = "_".join(attributs)
    #     ld = Loader()
    #     nbr_itr_per_epoch = int(len(self.image_path)/batch_size)

    #     if weights == "":
    #         info('Compiling Model')
    #         self.gan.compile()

    #     else:
    #         self.gan.built = True
    #         self.gan = self.gan.load_weights(weights)

    #     info('start training') 
    #     for epoch in range (epochs):
         
    #         for i in range (3):
    #             lambda_e = 0.0001 * (epoch*nbr_itr_per_epoch + i)/(nbr_itr_per_epoch*epochs)

    #             imgs, atts = ld.Load_Data(batch_size,i, attributs)
    #             #for img, att in zip(np.array(imgs), atts):
    #             loss_model, loss_diss, loss_ae = self.gan.train_step(imgs, atts, lambda_e)
    #             print(f'epoch : {epoch} ------ iteration : {i}  ------   model_loss : {loss_model} ---------- loss_dis : {loss_diss} ------- loss_ae : {loss_ae}')
    #             save_loss(f'epoch : {epoch} ------ iteration : {i}  ------   model_loss : {loss_model} ---------- loss_dis : {loss_diss} ------- loss_ae : {loss_ae}')
    #         # print(f'epoch : {epoch}  --------   loss = {loss_model}')

    #     self.gan.layers[1].save_weights(PATH + f"\\utils\\models\\{name_attributs}_{date.today().strftime('%d-%m-%Y_%Hh%M')}.h5")

    #     info(f"epoch: {epoch} finished OK")


    def training(self, epochs, batch_size, attributs):
        ld = Loader()
        nbr_itr_per_epoch = int(len(self.image_path)/batch_size)
        info('Compiling Model')
        self.gan.compile()

        info('start training') 
        for epoch in range (epochs):
         
            for i in range (nbr_itr_per_epoch):
                lambda_e = 0.0001 * (epoch*nbr_itr_per_epoch + i)/(nbr_itr_per_epoch*epochs)

                imgs, atts = ld.Load_Data(batch_size,i, attributs)
                #for img, att in zip(np.array(imgs), atts):
                loss_model, loss_diss, loss_ae = self.gan.train_step(imgs, atts, lambda_e)
                print(f'epoch : {epoch} ------ iteration : {i}  ------   model_loss : {loss_model} ---------- loss_dis : {loss_diss} ------- loss_ae : {loss_ae}')
                save_loss(f'epoch : {epoch} ------ iteration : {i}  ------   model_loss : {loss_model} ---------- loss_dis : {loss_diss} ------- loss_ae : {loss_ae}')
            # print(f'epoch : {epoch}  --------   loss = {loss_model}')

        info(f"epoch: {epoch} finished OK")







