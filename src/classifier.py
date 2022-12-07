import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense
from keras.layers import BatchNormalization, Activation
from keras.layers import LeakyReLU, Conv2D
import keras.layers 
from keras.models import Sequential, Model
from keras.optimizers import Adam
import os
import matplotlib as plt
from model import Encoder
import argparse
from train import Loader
from config import debug, info, warning, log_config, PATH, save_loss
import glob
import cv2

# def creat_encoder(x, IMG_SIZE = 256, hid_dim = 512, init_fm = 16, max_filter = 512):

#     nb_layers = int(np.log2(hid_dim/init_fm))
#     layer_filter = [init_fm]
#     for i in range (nb_layers):
#         layer_filter.append(2*layer_filter[-1])
#     #print(layer_filter)
#     input_layer =  tf.keras.layers.Conv2D(init_fm, (4,4),strides=(2,2),padding='same', input_shape = (IMG_SIZE, IMG_SIZE, 3))
#     hid_layer = []

#     for i in layer_filter[1:]:
#         hid_layer.append(tf.keras.layers.Conv2D(i, (4,4),strides=(2,2),padding='same'))

#     output_layer = tf.keras.layers.Conv2D(max_filter, (4,4),strides=(2,2),padding='same')

#     # Encoder Model

#     x = input_layer(x)
#     x=tf.keras.layers.BatchNormalization()(x)
#     #BatchNormalization()
#     x=tf.keras.layers.LeakyReLU(alpha=0.2)(x)
#     for i in range(nb_layers):
#         x = hid_layer[i](x)
#         x=tf.keras.layers.BatchNormalization()(x)
#         #BatchNormalization()
#         x=tf.keras.layers.LeakyReLU(alpha=0.2)(x)

#     x = output_layer(x)
#     x=tf.keras.layers.BatchNormalization()(x)
#     #BatchNormalization()
#     x=tf.keras.layers.LeakyReLU(alpha=0.2)(x)

#     return x 

class Classifier(Model):
    def __init__(self,attributs):

        self.shape = (256,256,3)
        self.nbr_attributes = len(attributs)
        self.image_path = glob.glob(PATH + '\\data\\train' + '\\*.jpg')
        self.attributs = attributs
        

        super(Classifier,self).__init__()

        # self.shape = input_shape
        self.encoder = Encoder()
        self.conv2d_layer = tf.keras.layers.Conv2D(512, 4, 2, 'same', activation=LeakyReLU(0.2)) 
        self.bach_normalisation = tf.keras.layers.BatchNormalization()
        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense_layer1 = tf.keras.layers.Dense(512, activation = LeakyReLU(0.2))
        self.dense_layer2 = tf.keras.layers.Dense(self.nbr_attributes,activation='sigmoid')

        self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(lr = 0.0002)

    # def call(self,training=None, **kwargs):

    #     '''
    #     Model forward pass, when we use our model
    #     args:
    #         inputs : Model inputs (images)
    #     return:
    #         x : Output of the model (the attributes)
    #     '''
    #     # x = self.layers[0](inputs)
    #     # x=BatchNormalization()(x)
    #     # x=LeakyReLU(alpha=0.2)(x)
    #     # for i in (1,self.nb_layers):
    #     #     x = self.layers[i](x)
    #     #     x=BatchNormalization()(x)
    #     #     x=LeakyReLU(alpha=0.2)(x)

    #     # x = self.layers[-2](x)
    #     # x=LeakyReLU(alpha=0.2)(x)
    #     # x = self.layers[-1](x)
    #     # inputs = tf.keras.layers.Input(shape=self.shape)
    #     # x =  self.classif()(inputs)
    #     # x =  self.conv2d_layer()(x)
    #     # DNN = tf.keras.models.Model(inputs,x,name='classifier')

    #     # return DNN

    def creat_model(self):
            
        inputs = tf.keras.layers.Input(shape=self.shape)
        x = self.encoder(inputs)
        x = self.conv2d_layer(x)
        x = self.bach_normalisation(x)
        # x = self.flatten_layer(x)
        x = self.dense_layer1(x)
        x = self.dense_layer2(x)
        x = tf.keras.layers.Reshape((self.nbr_attributes,))(x)
        DNN = tf.keras.models.Model(inputs,x,name='classifier')
        DNN.build((None, 256, 256, 3))

        return DNN


    # def compile(self, 
    #             Optimizer = Adam(lr = 0.0002), 
    #             loss_function = keras.losses.CategoricalCrossentropy() ):

    #     super(Classifier, self).compile()
    #     #self.model.compile(optimizer=Optimizer, loss=loss_function)
    #     self.c_optimizer   = Optimizer
    #     self.loss          = loss_function

    @property
    def metrics(self):
        return self.c_loss_metric
    
    def train_step(self, images,attributs):

        # inputs is our batch images
        # batch_size=tf.shape(images)[0]
        print(images.shape)
        with tf.GradientTape() as tape:
            predictions = self.creat_model()(images)
            print('prediction',predictions)
            print('attributs',attributs)
            c_loss = self.loss_fn(attributs,predictions)

        #Backward
        grads = tape.gradient(c_loss, self.creat_model.trainable_weights)
        self.optimizer.apply_gradients( zip(grads, self.creat_model.trainable_weights) )

        self.c_loss.update_state(c_loss)

        return {"c_loss": self.c_loss_metric.result()}


    def training(self, epochs, batch_size):
        name_attributs = "_".join(self.attributs)
        ld = Loader()
        nbr_itr_per_epoch = 200 #int(len(self.image_path)/batch_size)

        # info('start training') 
        for epoch in range (epochs):
         
            for i in range (nbr_itr_per_epoch):
                lambda_e = 0.0001 * (epoch*nbr_itr_per_epoch + i)/(nbr_itr_per_epoch*epochs)

                imgs, atts = ld.Load_Data(batch_size,i, self.attributs)
                self.train_step(imgs, atts)
        
        return 


    """
    this function need to be added in utils 

    def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):

        @cm : 

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    """
    # def reload(self,filename):
    #     filename, extension = os.path.splitext(filename)
    #     self.Classifier = keras.models.load_model(f'{filename}-classifier.h5', compile=False)
    #     print('Reloaded.')

    # def save(self,filename):
    #     save_dir             = os.path.dirname(filename)
    #     filename, _extension = os.path.splitext(filename)
    #     # ---- Create directory if needed
    #     os.makedirs(save_dir, mode=0o750, exist_ok=True)
    #     # ---- Save models
    #     self.classifier.save( f'{filename}-classifier.h5' )


# def training_loop(DNN, epoch,bach_size,):

#         # inputs is our batch images
#         # batch_size=tf.shape(images)[0]

#         with tf.GradientTape() as tape:
#             predictions = self.model(images)
#             c_loss = self.loss_fn(attributs,predictions)

#         #Backward
#         grads = tape.gradient(c_loss, self.model.trainable_weights)
#         self.optimizer.apply_gradients( zip(grads, self.model.trainable_weights) )

#         self.c_loss.update_state(c_loss)

#         return {"c_loss": self.c_loss_metric.result()}





if __name__ == '__main__':


    # parser = argparse.ArgumentParser(description='Classifier')
    # parser.add_argument("--img_path", type = str, default = "data/img_align_celeba_resized", help= "Path to images")
    # parser.add_argument("--attr_path" ,type = str, default = "data/attributes.npz", help = "path to attributes")
    # parser.add_argument("--batch_size", type = int, default = 32, help= "Size of the batch used during the training")
    # parser.add_argument("--attr", type = str, default= "*", help= "Considered attributes to train the network with")
    # parser.add_argument("--n_epoch", type = int, default = 5, help = "Numbers of epochs")
    # parser.add_argument("--epoch_size", type = int, default = 50000, help = "Number of images seen at each epoch")
    # parser.add_argument("--n_images", type = int, default = 202599, help = "Number of images")
    # shape = (256,256,3) 
    # cls = Classifier(shape)
    # # cls.compile()
    # cls.creat_model()
    # cls.summary()
    # print('hello')
    # print(os.getcwd())
    img = cv2.imread(os.getcwd()+'\data\\train\\000001.jpg')
    # print(img.shape)
    cls = Classifier(["Smiling"])
    # cls.creat_model()
    # cls.build(img.shape'
    # cls.summary()
    # print('DNN')
    DNN = cls.creat_model()
    # DNN.build(img.shape)
    DNN.summary()
    cls.training(3,32)

