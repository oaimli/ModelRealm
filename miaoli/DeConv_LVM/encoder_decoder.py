#!/usr/bin/python3
# -*- coding: utf-8 -*-

from __future__ import print_function

from keras.layers import *
from keras import backend as K


def encoder_net_rnn(x, data_characteristics):
    inputs_reshape = Reshape((data_characteristics["words_dim"], data_characteristics["embedding_dim"]))(x)
    h = Bidirectional(LSTM(1024, return_sequences=True, kernel_initializer="glorot_uniform"), merge_mode='concat')(
        inputs_reshape)
    h = Bidirectional(LSTM(1024, return_sequences=False, kernel_initializer="glorot_uniform"), merge_mode='concat')(h)
    return h


def encoder_net_cnn(x, data_characteristics):
    h = Reshape(target_shape=(data_characteristics["words_dim"], data_characteristics["embedding_dim"], 1))(x)

    h = BatchNormalization()(h)
    h = Conv2D(filters=500, kernel_size=(3, data_characteristics["embedding_dim"]),
               kernel_initializer="glorot_uniform", strides=2)(h)
    h = Dropout(0.5)(h)
    h = Activation("relu")(h)

    h = Reshape((K.int_shape(h)[1], K.int_shape(h)[3], 1))(h)
    h = BatchNormalization()(h)
    h = Conv2D(filters=1000, kernel_size=(3, K.int_shape(h)[2]),
               kernel_initializer="glorot_uniform", strides=2)(h)
    h = Dropout(0.5)(h)
    h = Activation("relu")(h)

    h = Reshape((K.int_shape(h)[1], K.int_shape(h)[3], 1))(h)
    h = BatchNormalization()(h)
    h = Conv2D(filters=600,
               kernel_size=(3, K.int_shape(h)[2]),
            kernel_initializer='glorot_uniform', strides=2)(h)
    h = Dropout(0.5)(h)
    h = Activation("relu")(h)

    h = Reshape((K.int_shape(h)[1], K.int_shape(h)[3], 1))(h)
    h = BatchNormalization()(h)
    h = Conv2D(filters=500,
               kernel_size=(3, K.int_shape(h)[2]),
               kernel_initializer='glorot_uniform', strides=1)(h)
    h = Dropout(0.5)(h)
    h = Activation("relu")(h)

    h = Flatten()(h)

    return h


def decoder_net_rnn(z, data_characteristics):
    # initial_state = Dense(1024)(z)
    zs = RepeatVector(data_characteristics["words_dim"])(z)
    outputs = LSTM(units=1024, return_sequences=True,
                   kernel_initializer="glorot_uniform")(zs)
    outputs = LSTM(units=1024, return_sequences=True,
                   kernel_initializer="glorot_uniform")(outputs)
    outputs = Concatenate()([outputs, zs])


    x_recon = TimeDistributed(Dense(data_characteristics["vocabulary_dim"], activation='softmax'))(outputs)

    # h = TimeDistributed(Dense(data_characteristics["embedding_dim"], activation='relu'))(outputs)
    # x_recon = Reshape((data_characteristics["words_dim"] * data_characteristics["embedding_dim"],))(outputs)
    return x_recon


def decoder_net_cnn(z, data_characteristics):
    h = Dense(8*500)(z)
    h = Dropout(0.5)(h)
    h = Reshape((8, 1, 500))(h)

    h = BatchNormalization()(h)
    h = Conv2DTranspose(filters=1, kernel_size=(3, 600), strides=1)(h)
    h = Activation("relu")(h)

    h = Reshape((K.int_shape(h)[1], 1, K.int_shape(h)[2]))(h)
    h = BatchNormalization()(h)
    h = Conv2DTranspose(filters=1, kernel_size=(3, 500), strides=2)(h)
    h = Activation("relu")(h)

    h = Reshape((K.int_shape(h)[1], 1, K.int_shape(h)[2]))(h)
    h = BatchNormalization()(h)
    h = Conv2DTranspose(filters=1, kernel_size=(3, 500), strides=2)(h)
    h = Activation("relu")(h)

    h = Reshape((K.int_shape(h)[1], 1, K.int_shape(h)[2]))(h)
    h = BatchNormalization()(h)
    h = Conv2DTranspose(filters=1, kernel_size=(3, 300), strides=2)(h)
    h = Activation("relu")(h)

    h = Reshape((K.int_shape(h)[1], K.int_shape(h)[2]))(h)
    x_recon = TimeDistributed(Dense(data_characteristics["vocabulary_dim"], activation='softmax'))(h)

    # x_recon = Reshape((data_characteristics["words_dim"] * data_characteristics["embedding_dim"],))(h)
    return x_recon