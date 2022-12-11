from train import Loader
import sys
import os
sys.append( os.path.dirname(os.path.abspath(__file__))[:-3] + "\\cfg")
from config import debug, info, warning, log_config, PATH
import glob
from model import Encoder, Decoder, Discriminator
log_config('test')
from fader_networks import GAN
import numpy as np


class Evaluation:
    def __init__ (self, attributes, batch_test, weights):
        self.test_ims_path = glob.glob(PATH + '\\data\\test' + '\\*.jpg')
        self.name_attributs = "_".join(self.attributs)
        self.weigths=weights
        self.ld = Loader()
        self.batch_test = batch_test
        
    def test(self):
        self.encoder, self.decoder, self.discriminator = Encoder(), Decoder(), Discriminator()
        self.encoder.load_weights(f'{self.weights}encoder/')
        info('Get weights of encoder')
        self.decoder.load_weights(f'{self.weights}decoder/')
        info('Get weights of decoder')
        self.discriminator.load_weights(f'{self.weights}discriminator/')
        info('Get weights of discriminator')

        self.gan = GAN(encoder=self.encoder, decoder=self.decoder, discriminator=self.discriminator)
        self.gan.compile()
        i = 0
        imgs, atts = self.ld.Load_Data(self.batch_test, i, self.attributs, mod='test')
        z, y_predict, x_reconstruct = self.gan.combine_model(imgs, atts) 

        return imgs, x_reconstruct

    
    def plot_testImg (self):
        imgs, x_reconstruct = self.test()
        imgs = (np.array(imgs)+1)*127.5
        x_reconstruct = (np.array(x_reconstruct)+1)*127.5
        for i in range(imgs.shape[0]):
            image=cv2.vconcat([imgs[i],x_reconstruct[i]])
            plt.imshow(image)
        #x= np.concatenate((imgs,x_reconstruct), axis = 0)