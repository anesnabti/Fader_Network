#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from keras.optimizers import Adam

from keras.layers import RandomFlip

from config import debug, info, warning, log_config

import random


#argument: un tableau des attributs et leur valeur à modifier 
class GAN():

    def __init__(self,num_attribut_modif,valeur):
 
        if all(num_attribut_modif>-1) and all(num_attribut_modif<40) and all(int(num_attribut_modif)==num_attribut_modif):
            if all(valeur>-1) and all(valeur<2) and all(valeur==int(valeur)): 
                self.num_attribut_modif=num_attribut_modif
                self.valeur=valeur

        self.lambda_e=0
        self.optimizer = Adam(0.002, 0.5)
        self.batch_size=32

        
    def augmented_data(self,x):
        if random.random() < 0.5:
            x= RandomFlip("horizontal")(x)
                                 
        #on peut également modifier contrast, luminosité, saturation
        #https://www.tensorflow.org/tutorials/images/data_augmentation
        
        return x
    
    # cette fonction permet de recupérer z
    def encoder(self, x):
    
        nb_filter=[16,32,64,128,256,512]
        
        
        for i in nb_filter:  
            x=Conv2D(i, (4,4),strides=(2,2),padding=(1,1))(x)
            x=BatchNormalization()(x)
            x=LeakyReLU(alpha=0.2)(x)
        
        x=Conv2D(512, (4,4),strides=(1,1),padding=(1,1))(x)
        x=BatchNormalization()(x)
        z=LeakyReLU(alpha=0.2)(x)
            
        return z
    
    def modification_y(self,y):
        for i in range(len(self.num_attribut_modif)):
            y[self.num_attribut_modif[i]]=self.valeur[i]
        return y
    
    #cette fonction permet de générer l'image avec le y voulu
    def decoder(self,z,y):
        y=self.modification_y(y)
        y=tf.keras.utils.to_categorical(y, num_classes=2)
        y=tf.transpose(tf.stack([y]*4),[1, 0, 2])

        xx=2
        yy=2
        y=tf.reshape(y, [-1, xx, yy, 2*40])
        
        z=tf.concat([z,y],3)
        z=Conv2DTranspose(512, (4,4),strides=(1,1),padding=(1,1))(z)
        z=BatchNormalization()(z)
        z=Activation('relu')(z)
        
        nb_filter=[512,256,128,64,32,16]
        
        for n in nb_filter:
            y= tf.concat([y]*4,axis=1)
            y=Reshape(y, [-1, xx, yy, 2*40])
            
            z=tf.concat([z,y],3)
            z=Conv2DTranspose(n, (4,4),strides=(2,2),padding=(1,1))(z)
            z=BatchNormalization()(z)
            z=Activation('relu')(z)
            z=Dropout(0.3)(z)
            
            xx=xx*2
            yy=yy*2

        x2=Conv2DTranspose(3, (4,4),strides=(2,2),padding=(1,1))(z)

        #vérifier que x2 est de dimension (256, 256)
        
        #valeur de l'image entre -1 et 1
        x2=tf.math.tanh(x2)
        
        return x2
    
    #cette fonction permet de recupérer le y à partir de z
    def discriminateur(self,z):
        
        z=Conv2DTranspose(512, (4,4),strides=(2,2),padding=(1,1))(z)
        z=BatchNormalization()(z)
        z=Activation('relu')(z)
        
        #il n'y pas de dropout dans le discriminateur pour le model original
        #d'après l'article, il permet d'augmenter de façon significatif les performances
        z = Dropout(rate=0.3)(z)
            
        z=Dense(512, input_shape=(512,), activation=None)(z)
        z=LeakyReLU(alpha=0.2)
        z=Dense(40, input_shape=(512,), activation=None)(z)
        
        #valeur de y entre 0 et 1
        prediction_y=tf.math.sigmoid(z)
        
        return prediction_y
        
    def loss_ae(self,x,x2):
        l=tf.squared_difference(x, x2)
        l=tf.reduce_sum(l,[1, 2, 3])
        l=tf.reduce_mean(l)
        return l
    
    def loss_dis(self,y,prediction_y):
        probabilite=1-tf.abs(y-prediction_y)
        #attention: log non défini en 0
        l=tf.log(probabilite + 1e-8)
        l=tf.reduce_sum(l,1)
        l=-tf.reduce_mean(l)
        return l
    
    #cette fonction permet d'améliorer la création d'images  
    def loss(self,x,x2,y,prediction_y):
        return self.loss_ae(self,x,x2)+self.lambda_e*self.loss_dis(self,y,1-prediction_y)
        
            
    # def train(self, epochs, batch_size,x, y):

    #     for epoch in range(epochs):
        
    #         self.lambda_e=self.lambda_e + 0.00001/500000
            
    #         x=self.augmented_data(self,x)
    #         y_modif=modification_y(self,y)
            
    #         model_gan=self.decoder(self.encoder(x),y_modif)
            
    #         model_gan.compile(loss=self.loss,optimizer=self.optimizer,metrics=['accuracy'])
            
    #         model_dis=self.discriminateur(self.encoder(x))
            
    #         model_dis.compile(loss=self.loss,optimizer=self.optimizer,metrics=['accuracy'])
            
    #         #model.fit(train,validation_data=val,epochs=epoch)

        
    #     return model_gan




# if __name__ == '__main__':
#     gan = GAN([6],[1])
#     model_gan=gan.train(epochs=30000, batch_size=32,x, y)
#     model_gan.predict(test)
            
        
