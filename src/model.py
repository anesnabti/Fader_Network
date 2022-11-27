
import numpy as np
import tensorflow as tf
import os
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation
from keras.layers import LeakyReLU, ReLU
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam


#from ..cfg.config import debug, info, warning, log_config


IMG_SIZE = 256
# revoie la boucle et verifier si on construit pas deux fois les layers
class Encoder(Model):
    def __init__(self,hid_dim = 512, init_fm = 16, max_filter = 512 ):
        super(Encoder, self).__init__()
        self.nb_layers = int(np.log2(hid_dim/init_fm))
        layer_filter = [init_fm]
        for i in range (self.nb_layers):
            layer_filter.append(2*layer_filter[-1])

        self.input_layer =  Conv2D(init_fm, (4,4),strides=(2,2),padding='same', input_shape = (IMG_SIZE, IMG_SIZE, 3))
        self.hid_layer = []

        for i in layer_filter[1:]:
            self.hid_layer.append(Conv2D(i, (4,4),strides=(2,2),padding='same'))

        self.output_layer = Conv2D(max_filter, (4,4),strides=(1,1),padding='same')
    def call(self, inputs, training=None, **kwargs):
        print(self.nb_layers)
        x = self.input_layer(inputs)
        x=BatchNormalization()(x)
        x=LeakyReLU(alpha=0.2)(x)
        for i in range(self.nb_layers):
            x = self.hid_layer[i](x)
            x=BatchNormalization()(x)
            x=LeakyReLU(alpha=0.2)(x)

        x = self.output_layer(x)
        x=BatchNormalization()(x)
        x=LeakyReLU(alpha=0.2)(x)
        return x



class Discriminator(Model):
    def __init__(self,n_attr = 1 ):
        super(Discriminator, self).__init__()
        self.input_shape = (2,2,512)
        self.layer1 = Conv2DTranspose(512,(4,4),strides=(2,2),padding='same', input_shape = self.input_shape)
        self.layer2 = Dense(512, input_shape=(512,), activation=None)
        self.layer3 = Dense(n_attr, input_shape=(512,), activation=None)
        
    def call(self, inputs, training=None, **kwargs):
        x = self.layer1(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(0.3)(x)
        x = self.layer2(x)
        x = LeakyReLU(alpha = 0.2)(x)
        x = self.layer2(x)
        x = Activation('sigmoid')(x)
        return x



class Decoder(Model):
    def __init__(self,nbr_attr = 1, img_fm = 3, max_filter = 512,init_fm = 16 ):
        super(Decoder, self).__init__()

        n_dec_out = img_fm
        self.nb_layers = int(np.log2(max_filter/init_fm))
        self.latent_dim = (2,2,max_filter + 2*nbr_attr)
        n_dec_in = init_fm + nbr_attr
        
        filter_layer = [max_filter]
        for i in range (self.nb_layers):
            filter_layer.append(filter_layer[-1]/2)

        self.input_layer = Conv2DTranspose(512, (4,4),strides=(1,1),padding='same', input_shape = self.latent_dim)
        self.hid_layer = []
        for i in filter_layer[1:]:
            self.hid_dim.append(Conv2DTranspose(i, (4,4),strides=(2,2),padding='same'))
        self.output_layer = Conv2DTranspose(img_fm, (4,4),strides=(2,2),padding='same')

        def call(self, inputs, training=None, **kwargs):

            x = self.input_layer(inputs)
            x=BatchNormalization()(x)
            x=ReLU()(x)
            for i in range(self.nb_layer):
                x = self.hid_layer(x)
                x=BatchNormalization()(x)
                x=ReLU()(x)
                x=Dropout(0.3)(x)
            x = self.output_layer(x)
            #vérifier que x2 est de dimension (256, 256)

            #valeur de l'image entre -1 et 1
            x = tf.math.tanh(x)        # a verifier 
            return x

# #################  revoir  ##################
def input_decode(y,z):
    y=tf.keras.utils.to_categorical(y, num_classes=2)
    y=tf.transpose(tf.stack([y]*4),[1, 0, 2])

    xx=2
    yy=2
    y=tf.reshape(y, [-1, xx, yy, 2*40])
    
    z=tf.concat([z,y],3)
################################################

# Simple Autoencoder
class AutoEncoder(Model):
    def __init__ (self, encoder, decoder):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def call(self, inputs):
        '''
        Model forward pass, when we use our model
        args:
            inputs : Model inputs
        return:
            x_reconstruct : Output of the model 
        '''
        z = self.encoder(inputs)
        x_reconstruct = self.decoder(z)
        return x_reconstruct

    def train_step(self, input):
        '''
        Implementation of the training update.
        Receive an input, compute loss, get gradient, update weights and return metrics.
        Here, our metrics is the loss reconstruction.
        args:
            inputs : Model inputs
        return:
            r_loss  : Reconstruction loss
            
        '''
        
        # ---- Get the input we need, specified in the .fit()
        #
        if isinstance(input, tuple):
            input = input[0]
        
        #
        with tf.GradientTape() as tape:
            
            # ---- Get encoder outputs
            
            z = self.encoder(input)
            
            # ---- Get reconstruction from decoder
            #
            reconstruction       = self.decoder(z)
         
            # ---- Compute loss
            #
            reconstruction_loss  = tf.keras.losses.binary_crossentropy(input, reconstruction)

        grads = tape.gradient(reconstruction_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        return {
            "r_loss":   reconstruction_loss,
      
        }


    def save(self,filename):
        '''Save model'''
        filename, extension = os.path.splitext(filename)
        self.encoder.save(f'{filename}-encoder.h5')
        self.decoder.save(f'{filename}-decoder.h5')




if __name__ == '__main__':

    tmp = np.load('D:\M2\ML\Projet\Fader_N\Fader_Network\src\Fader_Network.npy')
    enc = Encoder()
    data  = tmp[0:10]
    # le rajout d'un reshape (-1, 256,256,3) n'est obligatoire que lorsqu'on donne une suele image à model
    data = data.reshape(-1, 256,256,3)
    print(data.shape)
    #enc.compile()
    print(enc(data))
    print(enc.get_weights())
    