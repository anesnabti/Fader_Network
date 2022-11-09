import cv2
import numpy as np
import matplotlib.image as mpimg
import pathlib
import os
import sys
from csv import writer
import tensorflow as tf
PARENT_PATH = str(pathlib.Path().parent.resolve())
sys.path.append(PARENT_PATH + "\\cfg")
from config import info, debug, warning, log_config

"""
This file aims to us to preprocess all images in our data. 
The purpose is to crop images and resize them to (256 x 256)
"""

PATH = PARENT_PATH + "\\data"
Nbr_images = 202599
SIZE_IMG = 256

log_config("preprocessing_images")

def preprocessing_images ():

    #verifying if all images are in data
    if len(os.listdir(PATH + "\\img_align_celeba")) != Nbr_images:
        debug("You do not have all images ! Please Check")
        return
    else: 
        info("OK : All images are downloded")

    print("############ Reading images ##############")
    read_img = []
    for i in range (1, Nbr_images + 1) :
        if i % 10000 == 0:
            print('iteration :',i)
        read_img.append(mpimg.imread(PATH + "\\img_align_celeba\\%06i.jpg" % i)[20:-20])     # %06i% means that we have a number of 6 digits | we do [20 : -20] to crop images into 178 x 178

    if len(read_img) != Nbr_images:
        debug(f"Found {len(read_img)}, must have {Nbr_images} ")
        raise Exception (f"Found {len(read_img)}, must have {Nbr_images} ")
    else: 
        info("All images have been read")

    # Resizing images : 
    print('############# Resizing Images #############')
    resize_img = []
    for i,img in enumerate(read_img): 
        if i % 10000 == 0:
            print('iteration :',i)

        if img.shape != (178,178,3):
            debug("Error cropped image")
            raise Exception (" Image %06i%  .does not been cropped correctly % i")

        img = cv2.resize(img, (SIZE_IMG, SIZE_IMG), interpolation=cv2.INTER_LANCZOS4)
        assert img.shape == (SIZE_IMG, SIZE_IMG, 3)
        resize_img.append(img)

    images = np.array(resize_img)
    info("All images are resized")
    data_images = tf.convert_to_tensor(images, np.float32)
    assert data_images.shape == (Nbr_images, SIZE_IMG, SIZE_IMG, 3)

    # Saving this image
    try : 
        np.save(PATH + "\\IMAGES", data_images)
        info("Model saved correctly")
    except : 
       debug("Images are not saved")

    
def preprocessing_labels():

    dataset_table = f'{os.getcwd()}/data/Anno/list_attr_celeba.txt'
    attr_lines = [line.rstrip() for line in open(dataset_table, 'r')]
    attr_keys = 'file_name' + ' '+ attr_lines[1]
    matdata = []
    for i in range(1,Nbr_images):
        # Add the header
        if i == 1:
            list_data = np.array(attr_keys.replace('  ',' ').replace(',','').split()).reshape(1,-1)[0]
            info("Header added correctly")
        # Add the attributs values
        else:
            list_data = np.array(attr_lines[i].replace('-1','0').replace('  ',' ').replace(',','').split()).reshape(1,-1)[0]
        matdata.append(list_data)
        info("Attributes values added correctly")

    # to save as npy

    if matdata.shape != (202598,41):
        debug(f"Found {matdata.shape}, must have {(202598,41)} ")
        raise Exception (f"Found {matdata.shape}, must have {(202598,41)} ")
        
    np.save(f'{os.getcwd()}/data/ATTRIBUTS.npy',np.array(matdata))
    info("ATTRIBUTES.npy saved correctly")
    
    # to save as csv
    with open('./data/list_attr_celebatest.csv', 'a', newline='') as f_object:  
        # Pass the CSV  file object to the writer() function
        writer_object = writer(f_object)
        # Result - a writer object
        # Pass the data in the list as an argument into the writerow() function
        writer_object.writerow(list_data)  
        # Close the file object
        f_object.close()
    info("ATTRIBUTES.csv saved correctly")



if __name__ == '__main__':
    preprocessing_images()
    preprocessing_labels()


