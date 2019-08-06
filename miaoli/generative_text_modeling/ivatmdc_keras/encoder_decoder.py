#!/usr/bin/python3
# -*- coding: utf-8 -*-

from __future__ import print_function

from keras.layers import *
from keras import backend as K


def encoder_net_rnn(x, data_characteristics):
    inputs_reshape = Reshape((data_characteristics["words_dim"], data_characteristics["embedding_dim"]))(x)
    h = Bidirectional(LSTM(1024, return_sequences=False, kernel_initializer="glorot_uniform"), merge_mode='concat')(
        inputs_reshape)
    return h


def encoder_net_cnn(x, data_characteristics):
    h = Reshape(target_shape=(data_characteristics["words_dim"], data_characteristics["embedding_dim"], 1))(x)

    h = Conv2D(filters=300, kernel_size=(5, data_characteristics["embedding_dim"]),
               kernel_initializer="glorot_uniform", strides=2)(h)
    h = Activation("relu")(h)

    h = Reshape((K.int_shape(h)[1], K.int_shape(h)[3], 1))(h)
    h = Conv2D(filters=600, kernel_size=(5, K.int_shape(h)[2]),
               kernel_initializer="glorot_uniform", strides=2)(h)
    h = Activation("relu")(h)

    h = Reshape((K.int_shape(h)[1], K.int_shape(h)[3], 1))(h)
    h = Conv2D(filters=500,
               kernel_size=(5, K.int_shape(h)[2]),
            kernel_initializer='glorot_uniform', strides=2)(h)
    h = Activation("relu")(h)

    h = Reshape((K.int_shape(h)[1], K.int_shape(h)[3], 1))(h)
    h = Conv2D(filters=600,
               kernel_size=(5, K.int_shape(h)[2]),
               kernel_initializer='glorot_uniform', strides=2)(h)
    h = Activation("relu")(h)

    h = Reshape((K.int_shape(h)[1], K.int_shape(h)[3], 1))(h)
    h = Conv2D(filters=500,
               kernel_size=(5, K.int_shape(h)[2]),
               kernel_initializer='glorot_uniform', strides=2)(h)
    h = Activation("relu")(h)

    h = Flatten()(h)

    return h


def decoder_net_rnn(z, data_characteristics):
    # initial_state = Dense(1024)(z)
    zs = RepeatVector(data_characteristics["words_dim"])(z)
    outputs = LSTM(units=1024, return_sequences=True,
                   kernel_initializer="glorot_uniform", dropout=0.5, recurrent_dropout=0.5)(zs)
    outputs = Concatenate()([outputs, zs])


    x_recon = TimeDistributed(Dense(data_characteristics["vocabulary_dim"], activation='softmax'))(outputs)

    # h = TimeDistributed(Dense(data_characteristics["embedding_dim"], activation='relu'))(outputs)
    # x_recon = Reshape((data_characteristics["words_dim"] * data_characteristics["embedding_dim"],))(outputs)
    return x_recon


def decoder_net_cnn(z, data_characteristics):

    h = Reshape((1, 500, 1))(z)

    # h = Reshape((K.int_shape(h)[1], K.int_shape(h)[2]))(h)
    # zs = RepeatVector(K.int_shape(h)[1])(z)
    # h = Concatenate()([h, zs])

    h = Reshape((K.int_shape(h)[1], 1, K.int_shape(h)[2]))(h)
    h = Conv2DTranspose(filters=1, kernel_size=(5, 500), strides=2)(h)
    h = Activation("relu")(h)

    h = Reshape((K.int_shape(h)[1], K.int_shape(h)[2]))(h)
    zs = RepeatVector(K.int_shape(h)[1])(z)
    h = Concatenate()([h, zs])

    h = Reshape((K.int_shape(h)[1], 1, K.int_shape(h)[2]))(h)
    h = Conv2DTranspose(filters=1, kernel_size=(5, 600), strides=2)(h)
    h = Activation("relu")(h)

    h = Reshape((K.int_shape(h)[1], K.int_shape(h)[2]))(h)
    zs = RepeatVector(K.int_shape(h)[1])(z)
    h = Concatenate()([h, zs])

    h = Reshape((K.int_shape(h)[1], 1, K.int_shape(h)[2]))(h)
    h = Conv2DTranspose(filters=1, kernel_size=(5, 500), strides=2)(h)
    h = Activation("relu")(h)

    h = Reshape((K.int_shape(h)[1], K.int_shape(h)[2]))(h)
    zs = RepeatVector(K.int_shape(h)[1])(z)
    h = Concatenate()([h, zs])

    h = Reshape((K.int_shape(h)[1], 1, K.int_shape(h)[2]))(h)
    h = Conv2DTranspose(filters=1, kernel_size=(5, 600), strides=2)(h)
    h = Activation("relu")(h)

    h = Reshape((K.int_shape(h)[1], K.int_shape(h)[2]))(h)
    zs = RepeatVector(K.int_shape(h)[1])(z)
    h = Concatenate()([h, zs])

    h = Reshape((K.int_shape(h)[1], 1, K.int_shape(h)[2]))(h)
    h = Conv2DTranspose(filters=1, kernel_size=(5, 300), strides=2)(h)
    h = Activation("relu")(h)

    h = Reshape((K.int_shape(h)[1], K.int_shape(h)[2]))(h)
    zs = RepeatVector(data_characteristics["words_dim"])(z)
    h = Concatenate()([h, zs])
    x_recon = TimeDistributed(Dense(data_characteristics["vocabulary_dim"], activation='softmax'))(h)

    # x_recon = Reshape((data_characteristics["words_dim"] * data_characteristics["embedding_dim"],))(h)
    return x_recon


def encoder_net_hybrid(x, data_characteristics):
    inputs_reshape = Reshape((data_characteristics["words_dim"], data_characteristics["embedding_dim"]))(x)
    h = Bidirectional(LSTM(1024, return_sequences=True, kernel_initializer="glorot_uniform"), merge_mode='concat')(
        inputs_reshape)

    h = Reshape(target_shape=(data_characteristics["words_dim"], 2048, 1))(h)

    h = BatchNormalization()(h)
    h = Conv2D(filters=1000, kernel_size=(3, 2048),
               kernel_initializer="glorot_uniform", strides=2)(h)
    h = Dropout(0.5)(h)
    h = Activation("relu")(h)

    h = Reshape((K.int_shape(h)[1], K.int_shape(h)[3], 1))(h)
    h = BatchNormalization()(h)
    h = Conv2D(filters=500, kernel_size=(3, K.int_shape(h)[2]),
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


def decoder_net_hybrid(z, data_characteristics):
    h = Dense(8 * 500)(z)
    h = Dropout(0.5)(h)
    h = Reshape((8, 1, 500))(h)

    h = BatchNormalization()(h)
    h = Conv2DTranspose(filters=1, kernel_size=(3, 600), strides=1)(h)
    h = Dropout(0.5)(h)
    h = Activation("relu")(h)

    h = Reshape((K.int_shape(h)[1], 1, K.int_shape(h)[2]))(h)
    h = BatchNormalization()(h)
    h = Conv2DTranspose(filters=1, kernel_size=(3, 500), strides=2)(h)
    h = Dropout(0.5)(h)
    h = Activation("relu")(h)

    h = Reshape((K.int_shape(h)[1], 1, K.int_shape(h)[2]))(h)
    h = BatchNormalization()(h)
    h = Conv2DTranspose(filters=1, kernel_size=(3, 1000), strides=2)(h)
    h = Dropout(0.5)(h)
    h = Activation("relu")(h)

    h = Reshape((K.int_shape(h)[1], 1, K.int_shape(h)[2]))(h)
    h = BatchNormalization()(h)
    h = Conv2DTranspose(filters=1, kernel_size=(3, 1000), strides=2)(h)
    h = Dropout(0.5)(h)
    h = Activation("relu")(h)

    h = Reshape((K.int_shape(h)[1], K.int_shape(h)[2]))(h)
    zs = RepeatVector(data_characteristics["words_dim"])(z)
    h = Concatenate()([h, zs])
    h = LSTM(units=1024, return_sequences=True,
                   kernel_initializer="glorot_uniform", dropout=0.5, recurrent_dropout=0.5)(h)
    h = Concatenate()([h, zs])
    x_recon = TimeDistributed(Dense(data_characteristics["vocabulary_dim"], activation='softmax'))(h)

    return x_recon
