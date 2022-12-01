import tensorflow as tf
import numpy as np
import cv2
import os 
import sys
import pathlib
from datetime import datetime as date
from load_data import Loader
from fader_networks import GAN
from model import Encoder, Decoder, Discriminator
sys.path.append( os.path.dirname(os.path.abspath(__file__))[:-3] + "\\cfg")
from config import debug, info, warning, log_config, PATH, save_loss
import glob
import matplotlib.pyplot as plt
Nbr_images = 202599
log_config('train')
image_path = glob.glob(PATH + '\\data\\train' + '\\*.jpg')
from fader_networks import GAN
import math

def plot_images(x,y=None, indices='all', columns=12, x_size=1, y_size=1,
                cm='binary',y_padding=0.35, spines_alpha=1,
                fontsize=20, save_as='auto'):
    """
    Plot original images
    Show some images in a grid, with legends
    args:
        x             : images - Shapes must be (-1,lx,ly) (-1,lx,ly,1) or (-1,lx,ly,3)
        y             : real classes or labels or None (None)
        indices       : indices of images to show or None for all (None)
        columns       : number of columns (12)
        x_size,y_size : figure size (1), (1)
        cm            : Matplotlib color map (binary)
        y_padding     : Padding / rows (0.35)
        font_size     : Font size in px (20)
        save_as       : Filename to use if save figs is enable ('auto')
    """
    if indices=='all': indices=range(len(x))
    draw_labels = (y is not None)
    rows        = math.ceil(len(indices)/columns)
    fig=plt.figure(figsize=(columns*x_size, rows*(y_size+y_padding)))
    n=1
    for i in indices:
        axs=fig.add_subplot(rows, columns, n)
        n+=1
        # ---- Shape is (lx,ly)
        if len(x[i].shape)==2:
            xx=x[i]
        # ---- Shape is (lx,ly,n)
        if len(x[i].shape)==3:
            (lx,ly,lz)=x[i].shape
            if lz==1: 
                xx=x[i].reshape(lx,ly)
            else:
                xx=x[i]
        img=axs.imshow(xx,   cmap = cm, interpolation='lanczos')
        """
        axs.spines['right'].set_visible(True)
        axs.spines['left'].set_visible(True)
        axs.spines['top'].set_visible(True)
        axs.spines['bottom'].set_visible(True)
        axs.spines['right'].set_alpha(spines_alpha)
        axs.spines['left'].set_alpha(spines_alpha)
        axs.spines['top'].set_alpha(spines_alpha)
        axs.spines['bottom'].set_alpha(spines_alpha)
        """
        
        for spine in ['bottom', 'left','top','right']:
            axs.spines[spine].set_visible(False)
        
        axs.set_yticks([])
        axs.set_xticks([])

        if draw_labels:
            axs.set_xlabel(y[n-2],fontsize=fontsize/2)

    # a de-commenter pour enregistrer les images generer par notre model 
    plt.savefig(PATH + "\\utils\\result_train\\im1.png")
    # cv2.imwrite(PATH + "\\utils\\result_train\\im2.png", np.array(xx))
    # plt.show()



class Train:

    def __init__ (self, lr, attributs):
 
        self.image_path = glob.glob(PATH + '\\data\\train' + '\\*.jpg')
        self.lr = lr
        self.attributs = attributs
        self.nbr_attr = len(attributs)
        self.gan = GAN(encoder=Encoder(), decoder=Decoder(nbr_attr = self.nbr_attr,disc = True), discriminator=Discriminator(self.nbr_attr))


    def result_train(self, real_image, reconstruct_image, epochs, nbr_itr_epoch, nbr_img=5):
        real_image = np.array(real_image)* 127.5 + 1
        reconstruct_image = np.array(reconstruct_image)*127.5 + 1
        I = []
        for i in range (nbr_img):
            I.append(real_image[i])
            for j in range (epochs):
                I.append(reconstruct_image[j*(nbr_itr_epoch) + i])


        plot_images(np.array(I),columns=epochs+1)






    def training(self, epochs, batch_size, weights=""):
        name_attributs = "_".join(self.attributs)
        ld = Loader()
        nbr_itr_per_epoch = 5 #int(len(self.image_path)/batch_size)
        real_image = []
        reconstruct_image = []
        info('Compiling Model')
        if weights =="":
            self.gan.compile()
        else:
            self.gan = self.gan.load_weights(weights)


        info('start training') 
        for epoch in range (epochs):
         
            for i in range (nbr_itr_per_epoch):
                lambda_e = 0.0001 * (epoch*nbr_itr_per_epoch + i)/(nbr_itr_per_epoch*epochs)

                imgs, atts = ld.Load_Data(batch_size,i, self.attributs)
                loss_model, loss_diss, loss_ae, x_reconstruct = self.gan.train_step(imgs, atts, lambda_e)
                print(f'epoch : {epoch} ------ iteration : {i}  ------   model_loss : {loss_model} ---------- loss_dis : {loss_diss} ------- loss_ae : {loss_ae}')
                info(f'epoch : {epoch} ------ iteration : {i}  ------   model_loss : {loss_model} ---------- loss_dis : {loss_diss} ------- loss_ae : {loss_ae}')
                real_image.append(imgs[-1])
                reconstruct_image.append(x_reconstruct[-1])


        self.result_train(real_image, reconstruct_image, epochs, nbr_itr_per_epoch)
       
        if not os.path.exists(PATH + f"\\utils\\models\\Model_{name_attributs}_{date.today().strftime('%d-%m-%Y')}"):
            os.makedirs(PATH + f"\\utils\\models\\Model_{name_attributs}_{date.today().strftime('%d-%m-%Y')}")
        
        self.gan.save_weights(PATH + f"\\utils\\models\\Model_{name_attributs}_{date.today().strftime('%d-%m-%Y')}\\")

        info(f"epoch: {epoch} finished OK")


    # def training(self, epochs, batch_size, attributs):
    #     ld = Loader()
    #     nbr_itr_per_epoch = int(len(self.image_path)/batch_size)
    #     info('Compiling Model')
    #     self.gan.compile()

    #     info('start training') 
    #     for epoch in range (epochs):
         
    #         for i in range (nbr_itr_per_epoch):
    #             lambda_e = 0.0001 * (epoch*nbr_itr_per_epoch + i)/(nbr_itr_per_epoch*epochs)

    #             imgs, atts = ld.Load_Data(batch_size,i, attributs)
    #             #for img, att in zip(np.array(imgs), atts):
    #             loss_model, loss_diss, loss_ae = self.gan.train_step(imgs, atts, lambda_e)
    #             print(f'epoch : {epoch} ------ iteration : {i}  ------   model_loss : {loss_model} ---------- loss_dis : {loss_diss} ------- loss_ae : {loss_ae}')
    #             save_loss(f'epoch : {epoch} ------ iteration : {i}  ------   model_loss : {loss_model} ---------- loss_dis : {loss_diss} ------- loss_ae : {loss_ae}')
    #         # print(f'epoch : {epoch}  --------   loss = {loss_model}')

    #     info(f"epoch: {epoch} finished OK")







