
"""
Created on Mon Oct 31 09:35:49 2022

@author: bouinsalome
"""

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model
from model import encoder, decoder, discriminator, input_decode

class AutoEncoder(Model):
    def __init__ (self, encoder, decoder):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def call(self, img):
        z = self.encoder(img)
        x_reconstruct = self.decoder(z)


class GAN(Model):
    '''
    Our model, built from given encoder and decoder and discriminator
    '''
    def __init__(self, encoder=None, decoder=None, discriminator = None, **kwargs):
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
        self.optimizer = tf.keras.optimizers.Adam()
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
        z = self.encoder(img)
        y_predict = self.discriminator(z)
        z = input_decode(att, z)
        x_reconstruct = self.decoder(z)


        return z, y_predict, x_reconstruct

    def get_loss(self):
        loss_ae = tf.keras.losses.mse()
        loss_discrimintor = tf.keras.mse()

        return loss_ae, loss_discrimintor

    
    @tf.function
    def train_step(self, img, att, lamda_e):
        """
        Train the model for one step
        """
        loss_ae, loss_discriminator = self.get_loss()
        self.discriminator.trainable = True
        self.ae.trainable = False

        with tf.GradientTape() as Tape:
            z, y_predict, x_reconstruct = self.combine_model(img, att)
            
            loss_diss = loss_discriminator(att, y_predict)

        gradient_diss = Tape.gradient(loss_diss, self.discriminator.trainable_weights)
        self.optimizer.apply_gradients(zip(gradient_diss, self.discriminator.trainable_weights))
        
        self.discriminator.trainable = False
        self.ae.trainable = True
        with tf.GradientTape() as Tape:
            z, y_predict, x_reconstruct = self.combine_model(img, att)
            loss_reconstruct = loss_ae(img, x_reconstruct)
            loss_model = loss_reconstruct + lamda_e*loss_diss

        gradient_rec = Tape.gradient(loss_model, self.ae.trainable_weights)
        self.optimizer.apply_gradients(zip(gradient_rec, self.ae.trainable_weights))
        
        return loss_model
        







