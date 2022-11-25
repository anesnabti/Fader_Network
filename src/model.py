import numpy as np
import tensorflow as tf

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation
from keras.layers import LeakyReLU
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam

from keras.layers import RandomFlip

from ..cfg.config import debug, info, warning, log_config

import random

IMG_SIZE = 256

def encoder(hid_dim = 512, init_fm = 16, max_filter = 512):

    nb_layers = int(np.log2(hid_dim/init_fm))

    layer = [init_fm]
    for i in range (nb_layers+1):
        layer.append(2*layer[-1])
    
    # conv layers constructions
    #initialiser
    inputs    = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x=Conv2D(i, (4,4),strides=(2,2),padding=(1,1))(inputs)
    x=BatchNormalization()(x)
    x=LeakyReLU(alpha=0.2)(x)

    for i in layer:
        if i !=init_fm:
            x=Conv2D(i, (4,4),strides=(2,2),padding=(1,1))(x)
            x=BatchNormalization()(x)
            x=LeakyReLU(alpha=0.2)(x)
    
    x=Conv2D(max_filter, (4,4),strides=(1,1),padding=(1,1))(x)
    x=BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    encoder = Model(inputs, x , name = 'encoder')
    return x


"""
# revoir
def input_decode(y,z):
    y=tf.keras.utils.to_categorical(y, num_classes=2)
    y=tf.transpose(tf.stack([y]*4),[1, 0, 2])

    xx=2
    yy=2
    y=tf.reshape(y, [-1, xx, yy, 2*40])
    
    z=tf.concat([z,y],3)


def decoder(nbr_attr, max_filter = 512):
    latent_dim = (2,2,max_filter + 2*nbr_attr)
    inputs  = Input(shape=(latent_dim,))

    z=Conv2DTranspose(512, (4,4),strides=(1,1),padding=(1,1))(inputs)
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


    """

lambda_e=0
optimizer = Adam(0.002, 0.5)
batch_size=32

def modification_y(self,y):
    for i in range(len(self.num_attribut_modif)):
        y[self.num_attribut_modif[i]]=self.valeur[i]
    return y

def decoder(z,y):

    y=modification_y(y)
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
def discriminateur(z):
    
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
    
def loss_ae(x,x2):
    l=tf.squared_difference(x, x2)
    l=tf.reduce_sum(l,[1, 2, 3])
    l=tf.reduce_mean(l)
    return l

def loss_dis(y,prediction_y):
    probabilite=1-tf.abs(y-prediction_y)
    #attention: log non défini en 0
    l=tf.log(probabilite + 1e-8)
    l=tf.reduce_sum(l,1)
    l=-tf.reduce_mean(l)
    return l

#cette fonction permet d'améliorer la création d'images  
def loss(x,x2,y,prediction_y):
    return loss_ae(x,x2)+lambda_e*loss_dis(y,1-prediction_y)
    
        
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