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



class GAN(Model):
    '''
    Our model, built from given encoder and decoder and discriminator
    '''
    def __init__(self, encoder=None, decoder=None, discriminator = None, **kwargs):
        '''
        GAN instantiation with encoder, decoder and discriminator
        args :
            encoder : Encoder model
            decoder : Decoder model
            discriminator : Discriminator model
        return:
            None
        '''
        super(GAN, self).__init__(**kwargs)
        self.encoder       = encoder
        self.decoder       = decoder
        self.discriminator = discriminator
