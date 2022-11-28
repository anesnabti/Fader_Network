
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

class Classifier(Model):
    def __init__(self, params):
        super(Classifier,self).__init__()

        self.classif = Encoder()
        self.classif.add(Conv2D(512, 4, 2, 'same', activation=LeakyReLU(0.2)))




    def call(self, inputs, training=None, **kwargs):

        '''
        Model forward pass, when we use our model
        args:
            inputs : Model inputs (images)
        return:
            x : Output of the model (the attributes)
        '''
        x = self.layers[0](inputs)
        x=BatchNormalization()(x)
        x=LeakyReLU(alpha=0.2)(x)
        for i in (1,self.nb_layers):
            x = self.layers[i](x)
            x=BatchNormalization()(x)
            x=LeakyReLU(alpha=0.2)(x)

        x = self.layers[-2](x)
        x=LeakyReLU(alpha=0.2)(x)
        x = self.layers[-1](x)

        return x


    def compile(self, 
                Optimizer = Adam(lr = 0.0002), 
                loss_function = keras.losses.CategoricalCrossentropy() ):

        super(Classifier, self).compile()
        #self.model.compile(optimizer=Optimizer, loss=loss_function)
        self.c_optimizer   = Optimizer
        self.loss          = loss_function

    @property
    def metrics(self):
        return self.c_loss_metric
    
    def train_step(self, images,attributs):

        # inputs is our batch images
        batch_size=tf.shape(images)[0]

        with tf.GradientTape() as tape:
            predictions = self.model(images)
            c_loss = self.loss(attributs,predictions)

        #Backward
        grads = tape.gradient(c_loss, self.model.trainable_weights)
        self.d_optimizer.apply_gradients( zip(grads, self.model.trainable_weights) )

        self.c_loss.update_state(c_loss)

        return {
            "c_loss": self.c_loss_metric.result()}



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
    def reload(self,filename):
        filename, extension = os.path.splitext(filename)
        self.Classifier = keras.models.load_model(f'{filename}-classifier.h5', compile=False)
        print('Reloaded.')

    def save(self,filename):
        save_dir             = os.path.dirname(filename)
        filename, _extension = os.path.splitext(filename)
        # ---- Create directory if needed
        os.makedirs(save_dir, mode=0o750, exist_ok=True)
        # ---- Save models
        self.classifier.save( f'{filename}-classifier.h5' )


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Classifier')
    parser.add_argument("--img_path", type = str, default = "data/img_align_celeba_resized", help= "Path to images")
    parser.add_argument("--attr_path" ,type = str, default = "data/attributes.npz", help = "path to attributes")
    parser.add_argument("--batch_size", type = int, default = 32, help= "Size of the batch used during the training")
    parser.add_argument("--attr", type = str, default= "*", help= "Considered attributes to train the network with")
    parser.add_argument("--n_epoch", type = int, default = 5, help = "Numbers of epochs")
    parser.add_argument("--epoch_size", type = int, default = 50000, help = "Number of images seen at each epoch")
    parser.add_argument("--n_images", type = int, default = 202599, help = "Number of images")
    cls = Classifier()
    cls.compile()
    cls.summary()