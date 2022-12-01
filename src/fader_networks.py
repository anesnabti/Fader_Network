
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
        attr = 1 - att
        z_ = input_decode(z, attr)
        x_reconstruct = self.decoder(z_)


        return z, y_predict, x_reconstruct

    def get_loss(self):
        loss_ae = tf.keras.losses.MeanSquaredError()
        loss_discrimintor = tf.keras.losses.MeanSquaredError()                      # pttr à modifier

        return loss_ae, loss_discrimintor


    def compile(self):
        self.discriminator.compile(
            optimizer= tf.keras.optimizers.Adam(),
            loss = self.get_loss()[0])
        
        self.ae.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=self.get_loss()[1]
        )

    # def compile(self, weights=""):

    #     if weights =="":
    #         self.discriminator.compile(
    #             optimizer= tf.keras.optimizers.Adam(),
    #             loss = self.get_loss()[0])
            
    #         self.ae.compile(
    #             optimizer=tf.keras.optimizers.Adam(),
    #             loss=self.get_loss()[1]
    #         )

    #     else:
    #         # self.discriminator.built = True
    #         self.discriminator.load_weights(weights[0])
    #         # self.ae.built = True
    #         self.ae.load_weights(weights[1])
    

    # @tf.function
    def train_step(self, img, att, lamda_e):
        
        loss_ae, loss_discriminator = self.get_loss()
        self.discriminator.trainable = True
        self.ae.trainable = False

        with tf.GradientTape() as Tape:
            z, y_predict, x_reconstruct = self.combine_model(img, att)
            
            loss_diss = loss_discriminator(att, y_predict)
            loss_diss2 = 1 - loss_diss

        gradient_diss = Tape.gradient(loss_diss, self.discriminator.trainable_weights)
        self.discriminator.optimizer.apply_gradients(zip(gradient_diss, self.discriminator.trainable_weights))

        self.discriminator.trainable = False
        self.ae.trainable = True
        with tf.GradientTape() as Tape:
            z, y_predict, x_reconstruct = self.combine_model(img, att)
            flipt_attr = 1 - att
            loss_reconstruct = loss_ae(img, x_reconstruct)
            loss_model = loss_reconstruct + lamda_e*loss_discriminator(flipt_attr, y_predict)

        gradient_rec = Tape.gradient(loss_model, self.ae.trainable_weights)
        self.ae.optimizer.apply_gradients(zip(gradient_rec, self.ae.trainable_weights))
        
        return loss_model, loss_diss, loss_reconstruct, x_reconstruct
    







