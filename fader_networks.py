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



class GAN():
    
    def __init__(self):
     
         #self.train=load_dataset('train')
         #self.val=load_dataset('val')
         #self.test=load_dataset('test')
        
         self.optimizer = tf.keras.optimizers.Adam(0.002, 0.5)
         self.lambda_e=0
         
         #self.image=self.train.image
    
    
    
    def load_dataset(split):
         x_list=tf.data.Dataset.form_tensor_slices(np.load('./dataset/vae-celeba/{}.npy'.format(split)))
         print(x_list)
         return x_list
    
    '''
    @tf.function
    def normalize(image):
        self.image = (self.image - tf.reduce_min(self.image))/(tf.reduce_max(self.image) - tf.reduce_min(self.image))
        self.image = (2 * self.image) - 1
        return self.image
         
    @tf.function
    def reshape(self):
        self.image = tf.image.random_crop(self.image, (178, 178, 3))
        self.image = tf.image.resize(self.image, (256, 256))
        self.image = tf.image.random_flip_left_right(self.image,0.5)
        return self.image
    '''
    
    # cette fonction permet de recupérer z
    def encoder(self):
    
        model = Sequential()
        
        model.add(Conv2D(16, (4,4),strides=(2,2),padding=(1,1)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Conv2D(32, (4,4),strides=(2,2),padding=(1,1)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Conv2D(64, (4,4),strides=(2,2),padding=(1,1)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Conv2D(128, (4,4),strides=(2,2),padding=(1,1)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Conv2D(256, (4,4),strides=(2,2),padding=(1,1)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Conv2D(512, (4,4),strides=(2,2),padding=(1,1)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Conv2D(512, (4,4),strides=(2,2),padding=(1,1)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        
        return model
    
    #cette fonction permet de générer l'image avec le y voulu
    def decoder(self,model):
    
        #model = Sequential()
        
        #passer de 512 à 592 filtres ?
        
        model.add(Conv2DTranspose(592, (4,4),strides=(2,2),padding=(1,1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        
        model.add(Conv2DTranspose(592, (4,4),strides=(2,2),padding=(1,1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        
        model.add(Conv2DTranspose(336, (4,4),strides=(2,2),padding=(1,1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        
        model.add(Conv2DTranspose(208, (4,4),strides=(2,2),padding=(1,1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        
        model.add(Conv2DTranspose(144, (4,4),strides=(2,2),padding=(1,1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        
        model.add(Conv2DTranspose(112, (4,4),strides=(2,2),padding=(1,1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        
        model.add(Conv2DTranspose(96, (4,4),strides=(2,2),padding=(1,1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        
        #model.summary()
        
        return model
    
    #cette fonction permet de recupérer le y à partir de z
    #et dire si on arrive à detecter une vraie d'une fausse image
    #def loss_discriminator(self):
        
    #cette fonction permet d'améliorer la création d'images
    #def loss_adversarial(self):

    
    
    def train(self, epochs, batch_size):
    
        for epoch in range(epochs):
        
            self.lambda_e=self.lambda_e + 0.00001/500000
            
            model=self.decoder(self.encoder())
            
            model.compile(loss='binary_crossentropy',optimizer=self.optimizer,metrics=['accuracy'])
            #model.fit(train,validation_data=val,epochs=epoch)
            
            #tourne horizontalement les images avec unr probabilité de 0.5
        
        return model



if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=30000, batch_size=32)
    #model.predict(test)