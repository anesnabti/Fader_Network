import tensorflow as tf
import numpy as np
import cv2
import os 
import sys
from model import input_decode
import glob
sys.path.append(str(os.path.dirname(os.path.abspath(__file__)))[:-3] + "\\cfg")
from config import debug, info, warning, log_config, PATH
Nbr_images = 202599

""" 
This file will load the pre-processed data.
Split them on train, validation and test.
Normalize images
"""
attributes = np.load(str( os.path.dirname(os.path.abspath(__file__)))[:-3] + "data\\ATTRIBUTS.npy")
log_config("load_data")


class LoadData : 

    def __init__ (self, train_size, validation_size, test_size):
        if train_size < 0 or validation_size < 0 or test_size < 0 : 
            debug ("Split ratio dataset must be > 0 ")
            raise Exception ("Split ratio dataset must be > 0 ")
        if train_size + validation_size + test_size > 1:
            debug("Sum of split ratio must be equal to 1")
            raise Exception ("Sum of split ratio must be equal to 1")
        if train_size + validation_size + test_size != 1 :
            warning ("Sum of split ratio must be equal to 1")
        if validation_size == 0: 
            warning("Ratio validation is 0")

        self.train_size, self.validation_size, self.test_size = train_size, validation_size, test_size 



    def split_data (self):
        train_path = PATH + '\\data\\train'
        test_path = PATH + "\\data\\test"
        validation_path = PATH + "\\data\\validation"
        
        if not os.path.exists(train_path):
            os.makedirs(train_path)
        if not os.path.exists(test_path):
            os.makedirs(test_path)
        if not os.path.exists(validation_path):
            os.makedirs(validation_path)   
        
        ############ Cpying train images in the folder ...//data//train
        info("start copying train images in ...//data//train")
        print("start copying train images in ...//data//train")
        for i in range (1, int(Nbr_images*self.train_size) + 1):
            cv2.imwrite(train_path + "\\%06i.jpg" % i, cv2.imread(PATH + "\\data\\img_align_celeba_resize\\%06i.jpg" % i))
        info("Copying train images OK")
        
        ############## Copying test images into ..//data//test
        info("start copying test images in ...//data//test")
        print("start copying test images in ...//data//test")
        for i in range (int(Nbr_images*self.train_size) + 1, int(Nbr_images*self.train_size) + int(Nbr_images*self.test_size) + 1):
            cv2.imwrite(test_path + "\\%06i.jpg" % i, (cv2.imread(PATH + "\\data\\img_align_celeba_resize\\%06i.jpg" % i)))
        info("Copying test images OK")

        ############## Copying validation images in the folder ..//data//validation
        info("start copying validation images in ...//data//validation")
        print("start copying validation images in ...//data//validation")
        for i in range (int(Nbr_images*self.train_size) + int(Nbr_images*self.test_size) + 1,Nbr_images + 1):
            cv2.imwrite(validation_path + "\\%06i.jpg" % i, (cv2.imread(PATH + "\\data\\img_align_celeba_resize\\%06i.jpg" % i)))
        info("Copying validation images OK")

        attributes_title = []
        for i in range (len(attributes[0])):
            attributes_title.append(attributes[0][i])
        info("Copying first line of the table OK")


        info("start copying train images attributes in ...//data//train")
        print("start copying train images attributes in ...//data//train")
        train_images_attributes = []
        # train_images_attributes.append(attributes_title)
        for i in range (1, int(Nbr_images*self.train_size) + 1):
            list_att=[]
            for j in range (len(attributes[i])):
                list_att.append(attributes[i][j])
            train_images_attributes.append(list_att)
        np.save(train_path+'\\train_images_att.npy',np.array(train_images_attributes))
        info("Copying train images attributes OK")
        

        info("start copying test images attributes in ...//data//test")
        print("start copying test images attributes in ...//data//test")
        test_images_attributes = []
        # test_images_attributes.append(attributes_title)
        for i in range (int(Nbr_images*self.train_size) + 1, int(Nbr_images*self.train_size) + int(Nbr_images*self.test_size) + 1):
            list_att=[]
            for j in range (len(attributes[i])):
                list_att.append(attributes[i][j])
            test_images_attributes.append(list_att)
        np.save(test_path+'\\test_images_att.npy',np.array(test_images_attributes))
        info("Copying test images attributes OK")

        info("start copying validation images attributes in ...//data//validation")
        print("start copying validation images attributes in ...//data//validation")
        validation_images_attributes = []
        # validation_images_attributes.append(attributes_title)
        for i in range (int(Nbr_images*self.train_size) + int(Nbr_images*self.test_size) + 1, Nbr_images + 1):
            list_att=[]
            for j in range (len(attributes[i])):
                list_att.append(attributes[i][j])
            validation_images_attributes.append(list_att)
        np.save(validation_path+'\\validation_images_att.npy',np.array(validation_images_attributes))
        info("Copying validation images attributes OK")
        
        #nbr, nbr_att = len(images), len(attributes)
 
        # train_attributes = attributes[:int(self.train_size * nbr_att)]
        # validation_attributes = attributes[int(self.train_size * nbr_att) : int(self.validation_size * nbr_att)]
        # test_attributes = attributes[int(self.validation_size)* nbr_att : int(self.test_size) * nbr_att]
        # info("Split attributes OK")
        
        

# if __name__ == '__main__':
#     ld = LoadData(0.7, 0.15, 0.15)
#     ld.split_data()


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
        return image/127.5-1.0 



    def Load_Data(self, batch_size = 2,itr = 0, mod = 'train', attr = ['Smiling']):

        image_path = glob.glob(PATH + '\\data\\' + mod + '\\*.jpg')
        attributes_path = glob.glob(PATH + '\\data\\' + mod + '\\*.npy')
        attributes = np.load(attributes_path[0])
        ind = []
        for i in attr:
            ind.append(self.AVAILABLE_ATTR.index(i)+1)
        tmp_img = []
        tmp_attr = []
        
        for i in range (batch_size):
            tmp_img.append(self.normalize(cv2.imread(image_path[i + itr * batch_size])))
            tmp_attr.append(attributes[i + itr*batch_size][ind])
        return np.array(tmp_img), np.array(tmp_attr).astype(float)



# if __name__ == '__main__':
#     ld = Loader()
#     img, attr = ld.Load_Data()
#     print(attr.shape)
#     print(img.shape)
#     print(attr[0])
#     print(input_decode(attr))
