import tensorflow_probability as tfp
import tensorflow as tf
import tensorflow_addons as tfa

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import numpy as np
import scipy as sp
import seaborn as sns
import os
# import bcnn_utils as bcu
import cv2
import tqdm

sns.set_style('whitegrid')
sns.set_context('talk')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Conv2D, MaxPooling2D,
                                     Flatten, Dropout, DepthwiseConv2D,
                                     Activation, BatchNormalization, SpatialDropout2D)
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

import keras_tuner as kt
from keras import backend as K
from tensorflow.keras.saving import get_custom_objects

from sklearn.utils import class_weight

import utils as u

tfd = tfp.distributions
tfpl = tfp.layers


trimg_dir = "/mnt/g/intel_images/seg_train/"
tsimg_dir = "/mnt/g/intel_images/seg_test/"

trgen, tsgen = bu.img_gen(trimg_dir,tsimg_dir)

def swish(x):
    return (K.sigmoid(x)*x)

get_custom_objects().update({'swish':Activation(swish)})

def neg_loglike(ytrue, ypred):
    return -ypred.log_prob(ytrue)

def divergence(q,p,_):
    return tfd.kl_divergence(q,p)/14034

dropoff = 0.25
dropoff2 = 0.5

l1,l2 = 1e-3, 7e-4

def bmodel():

    mdl = Sequential(name='bsequential')

    # get_custom_objects().update({'swish':Activation(swish)})

    mdl.add(
        tf.keras.layers.Rescaling(1. / 255, input_shape=(150, 150, 3)
    ))

    # mdl.add(tf.keras.layers.Resizing(120,120))

    mdl.add(tf.keras.layers.RandomFlip("horizontal"))

    # mdl.add(tf.keras.layers.RandomRotation(0.45))


    mdl.add(
        tfpl.Convolution2DReparameterization(
            filters=32, kernel_size=(5, 5), padding='same',
            kernel_prior_fn=tfpl.default_multivariate_normal_fn,
            kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
            kernel_divergence_fn=divergence,
            bias_prior_fn=tfpl.default_multivariate_normal_fn,
            bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
            bias_divergence_fn=divergence,
            )
    )

    mdl.add(Activation("relu"))

    mdl.add(
        tfpl.Convolution2DReparameterization(
            filters=32, kernel_size=(5, 5), padding='same',
            kernel_prior_fn=tfpl.default_multivariate_normal_fn,
            kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
            kernel_divergence_fn=divergence,
            bias_prior_fn=tfpl.default_multivariate_normal_fn,
            bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
            bias_divergence_fn=divergence,
            )
    )

    # mdl.add(SpatialDropout2D(dropoff))
    mdl.add(Activation("relu"))

    mdl.add(
        tfpl.Convolution2DReparameterization(
            filters=32, kernel_size=(5, 5), padding='same',
            kernel_prior_fn=tfpl.default_multivariate_normal_fn,
            kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
            kernel_divergence_fn=divergence,
            bias_prior_fn=tfpl.default_multivariate_normal_fn,
            bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
            bias_divergence_fn=divergence,
            )
    )
    mdl.add(Activation("relu"))
    mdl.add(SpatialDropout2D(dropoff))

    mdl.add(MaxPooling2D(2,2))

    mdl.add(Conv2D(filters=64,
                   kernel_size=(3,3),
                   padding='same'
                  ))


    mdl.add(Activation("relu"))

    mdl.add(Conv2D(filters=64,
                   kernel_size=(3,3),
                   padding='same'
                  ))


    mdl.add(Activation("relu"))

    mdl.add(Conv2D(filters=64,
                   kernel_size=(3,3),
                   padding='same'
                  ))


    mdl.add(Activation("relu"))
    mdl.add(SpatialDropout2D(dropoff))


    mdl.add(MaxPooling2D(3,3))


    mdl.add(Conv2D(filters=128,
                   kernel_size=(3,3),
                   padding='same'
                  ))
    mdl.add(Activation("relu"))

    mdl.add(Conv2D(filters=128,
                   kernel_size=(3,3),
                   padding='same'
                  ))
    mdl.add(Activation("relu"))
    #
    mdl.add(Conv2D(filters=128,
                   kernel_size=(3,3),
                   padding='same'
                   ))
    mdl.add(Activation("relu"))
    mdl.add(SpatialDropout2D(dropoff))

    mdl.add(MaxPooling2D(2,2))

    mdl.add(Conv2D(filters=256,
                   kernel_size=(3,3),
                   padding='same'
                  ))
    mdl.add(Activation("relu"))

    mdl.add(Conv2D(filters=256,
                   kernel_size=(3,3),
                   padding='same'
                  ))
    mdl.add(Activation("relu"))
    #
    mdl.add(Conv2D(filters=256,
                   kernel_size=(1,1),
                   padding='same'
                   ))
    mdl.add(Activation("relu"))
    mdl.add(SpatialDropout2D(dropoff))

    mdl.add(MaxPooling2D(2,2))

    mdl.add(Conv2D(filters=512,
                   kernel_size=(3,3),
                   padding='same'
                  ))
    mdl.add(Activation("relu"))

    mdl.add(Conv2D(filters=512,
                   kernel_size=(3,3),
                   padding='same'
                  ))
    mdl.add(Activation("relu"))
    #
    mdl.add(Conv2D(filters=512,
                   kernel_size=(1,1),
                   padding='same'
                   ))
    mdl.add(Activation("relu"))
    mdl.add(SpatialDropout2D(dropoff))

    mdl.add(MaxPooling2D(3,3))
    mdl.add(Flatten())

    kernel_regs = tf.keras.regularizers.l1_l2(l1=l1, l2=l2)

    # mdl.add(Dense(1024,kernel_regularizer=kernel_regs))
    # mdl.add(Activation('relu'))
    # # mdl.add(Dropout(dropoff2))


    mdl.add(
        tfpl.DenseReparameterization(
        units=2048,
        activity_regularizer=kernel_regs,
        kernel_prior_fn=tfpl.default_multivariate_normal_fn,
        kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
        kernel_divergence_fn=divergence,
        bias_prior_fn=tfpl.default_multivariate_normal_fn,
        bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
        bias_divergence_fn=divergence,
        )
    )
    # mdl.add(Dense(1024,kernel_regularizer=kernel_regs))
    mdl.add(Activation('relu'))
    mdl.add(Dropout(dropoff2))

    # mdl.add(Dense(1024,kernel_regularizer=kernel_regs))
    mdl.add(
        tfpl.DenseReparameterization(
        units=1024,
        activity_regularizer=kernel_regs,
        kernel_prior_fn=tfpl.default_multivariate_normal_fn,
        kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
        kernel_divergence_fn=divergence,
        bias_prior_fn=tfpl.default_multivariate_normal_fn,
        bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
        bias_divergence_fn=divergence,
        )
    )
    mdl.add(Activation('relu'))
    mdl.add(Dropout(dropoff2))

    # mdl.add(
    #     tfpl.DenseReparameterization(
    #     units=256,
    #     activity_regularizer=kernel_regs,
    #     kernel_prior_fn=tfpl.default_multivariate_normal_fn,
    #     kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
    #     kernel_divergence_fn=divergence,
    #     bias_prior_fn=tfpl.default_multivariate_normal_fn,
    #     bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
    #     bias_divergence_fn=divergence,
    #     )
    # )
    # # mdl.add(Dense(1024,kernel_regularizer=kernel_regs))
    # mdl.add(Activation('relu'))
    # mdl.add(Dropout(dropoff2))

    mdl.add(
        tfpl.DenseReparameterization(
        units=tfpl.OneHotCategorical.params_size(6),
            kernel_prior_fn=tfpl.default_multivariate_normal_fn,
            kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
            kernel_divergence_fn=divergence,
            bias_prior_fn=tfpl.default_multivariate_normal_fn,
            bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
            bias_divergence_fn=divergence,
        )
    )


    mdl.add(tfpl.OneHotCategorical(6,
            convert_to_tensor_fn=tfd.Distribution.mode))
    return mdl


tf.keras.backend.clear_session()
bmdl = bmodel()
bmdl.summary()

earlystop = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    start_from_epoch=25,
    restore_best_weights=True,
    mode='max'
)


initial_learning_rate = 5e-3
lr_decay = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=2000,
    decay_rate=0.9,
    staircase=True)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',
    factor = 0.95,
    patience=5,
    cooldown=5,
    mode='max',
    min_lr=1e-7,
    verbose=1,
    min_delta=0.01)

bmdl.compile(
    loss=neg_loglike,
    optimizer=tf.keras.optimizers.Nadam(learning_rate=7e-5),
    metrics=['accuracy'],
    experimental_run_tf_function=False
)


bmdlhist = bmdl.fit(trgen,
                  # steps_per_epoch = 100,
                  # validation_steps = 50,
                  epochs=100, verbose=1,
                  validation_data=tsgen,
                  callbacks=[earlystop, reduce_lr])


export_path = "/mnt/g/WSL/models/"