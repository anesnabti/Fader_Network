
"""
Created on Mon Oct 31 09:35:49 2022

@author: bouinsalome
"""

import numpy as np
import tensorflow as tf

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation
from keras.layers import LeakyReLU
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from model import Encoder, Decoder, Discriminator,AutoEncoder, input_decode

"""
class AutoEncoder(Model):
    def __init__ (self, encoder, decoder):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def call(self, img):
        z = self.encoder(img)
        x_reconstruct = self.decoder(z)

"""

class GAN(Model):
    '''
    Our model, built from given encoder and decoder and discriminator
    '''
    def __init__(self, encoder=Encoder(), decoder=Decoder(), discriminator = Discriminator(), **kwargs):
        '''
        GAN instantiation with encoder, decoder and discriminator
        args :
            encoder : Encoder model
            decoder : Decoder model
            discriminator : Discriminator model
        return:
            None
        '''
        super(GAN, self).__init__(**kwargs)
        self.encoder       = encoder
        self.decoder       = decoder
        self.discriminator = discriminator
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
        self.ae = AutoEncoder(self.encoder, self.decoder)


    def combine_model(self,img, att):
        """ Args:
                img : image
                att: attribut
            return : 
                z : latent space
                y_predict: attribut predict for discriminator
                x_reconstruct: image reconstruct for decoder
        """
        #img = img.reshape(-1,256,256,3)
        z = self.encoder(img)
        y_predict = self.discriminator(z)
        z_ = input_decode(z, att)
        x_reconstruct = self.decoder(z_)


        return z, y_predict, x_reconstruct

    def get_loss(self):
        loss_ae = tf.keras.losses.MeanSquaredError()
        loss_discrimintor = tf.keras.losses.MeanSquaredError()

        return loss_ae, loss_discrimintor

    
    #@tf.function
    def train_step(self, img, att, lamda_e):
        
        loss_ae, loss_discriminator = self.get_loss()
        self.discriminator.trainable = True
        self.ae.trainable = False

        with tf.GradientTape() as Tape:
            z, y_predict, x_reconstruct = self.combine_model(img, att)
            
            loss_diss = loss_discriminator(att, y_predict)

        gradient_diss = Tape.gradient(loss_diss, self.discriminator.trainable_weights)
        #self.optimizer.apply_gradients(zip(gradient_diss, self.discriminator.trainable_weights))
        self.discriminator.compile(optimizer = self.optimizer, loss = tf.keras.losses.MeanSquaredError()).optimizer.apply_gradients(zip(gradient_diss, self.discriminator.trainable_weights))

        self.discriminator.trainable = False
        self.ae.trainable = True
        with tf.GradientTape() as Tape:
            z, y_predict, x_reconstruct = self.combine_model(img, att)
            loss_reconstruct = loss_ae(img, x_reconstruct)
            loss_model = loss_reconstruct + lamda_e*loss_diss

        gradient_rec = Tape.gradient(loss_model, self.ae.trainable_weights)
        self.optimizer.apply_gradients(zip(gradient_rec, self.ae.trainable_weights))
        
        return loss_model
    







