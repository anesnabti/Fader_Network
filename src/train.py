import tensorflow as tf
import numpy as np
import cv2
import os 
import sys
import pathlib
from datetime import datetime as date
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


class Loader():
    def __init__ (self):
# A revoir 
        self.AVAILABLE_ATTR = [
            "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
            "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair",
            "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair",
            "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache",
            "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose",
            "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair",
            "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
            "Wearing_Necklace", "Wearing_Necktie", "Young"
        ]

    def normalize (self,image):  
        return image/127.5 - 1 



    def Load_Data(self, batch_size, itr, attributs, mod = 'train'):

        image_path = glob.glob(PATH + '\\data\\' + mod + '\\*.jpg')
        attributes_path = glob.glob(PATH + '\\data\\' + mod + '\\*.npy')
        attributes = np.load(attributes_path[0])
        ind = []
        for i in attributs:
            ind.append(self.AVAILABLE_ATTR.index(i)+1)
        tmp_img = []
        tmp_attr = []
        
        for i in range (batch_size):
            tmp_img.append(self.normalize(cv2.imread(image_path[i + itr * batch_size])))
            tmp_attr.append(attributes[i + itr*batch_size][ind])
        return np.array(tmp_img), np.array(tmp_attr).astype(float)



class Train:

    def __init__ (self, lr, attributs):
 
        self.image_path = glob.glob(PATH + '\\data\\train' + '\\*.jpg')
        self.lr = lr
        self.attributs = attributs
        self.nbr_attr = len(attributs)
        self.gan = GAN(encoder=Encoder(), decoder=Decoder(nbr_attr = self.nbr_attr,disc = True), discriminator=Discriminator(self.nbr_attr))


    def result_train(self, real_image, reconstruct_image, epochs, nbr_itr_epoch, nbr_img=4):
        real_image = (np.array(real_image)+1)*127.5
        reconstruct_image = (np.array(reconstruct_image) + 1)*127.5 
        I, img_list = [], []
        for i in range (nbr_img):
            I.append(real_image[i])
            for j in range (epochs):
                I.append(reconstruct_image[j*(nbr_itr_epoch) + i])

        # print(type(I[1]))
        # exit()
        for i in range (int(len(I)/(epochs+1))):
            img_list.append(I[i*epochs + i : epochs*(i+1) + i + 1])

        img_list = np.array(img_list)
        
        
        im_list = [img_list[0],
                    img_list[1],
                    img_list[2],
                    img_list[3]]
        I_concat = cv2.vconcat([cv2.hconcat(im) for im in im_list])
        # plot_images(np.array(I),columns=epochs+1)
        cv2.imwrite(PATH + f"\\utils\\result_train\\train_img_{date.today().strftime('%d-%m-%Y-%Hh%M')}.png", I_concat)



    def training(self, epochs, batch_size):
        name_attributs = "_".join(self.attributs)
        ld = Loader()
        nbr_itr_per_epoch = 500 #int(len(self.image_path)/batch_size)
        real_image = []
        reconstruct_image = []
        self.gan.compile()

        # info('Compiling Model')
        # if weights ==" ":
        #     self.gan.compile()
        # else:
        #     self.gan = self.gan.load_weights(weights)


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

        info(f"epoch: {epoch + 1} finished OK")


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







