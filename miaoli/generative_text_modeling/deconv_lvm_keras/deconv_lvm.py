#! -*- coding: utf-8 -*-
# Text modeling using vae, to find valid encoder and decoder

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam

import numpy as np
import random

from sklearn import metrics
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

import sys
sys.path.append('../../../')
from miaoli.generative_text_modeling.utils.load_data import load_data
from miaoli.generative_text_modeling.utils.get_embeddings import generate_batch_train_indexes_mem_x, get_test_indexes_mem_vae
from miaoli.generative_text_modeling.utils.evaluation import plot_2d
from miaoli.generative_text_modeling.deconv_lvm_keras.encoder_decoder import encoder_net_cnn, decoder_net_cnn

# # specify GPU
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Not full of memory, on demand
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
KTF.set_session(sess)


# parameters
dataset = "20ng"
batch_size = 32
latent_dim = 500
epochs = 1000


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)

    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


# data load
train_set, test_set, word_embedding_dict, data_characteristics, word_index_dict, embedding_matrix = load_data(dataset, split_count=1000)
X_test, Y_test = get_test_indexes_mem_vae(random.sample(train_set, 2000), data_characteristics["words_dim"], word_index_dict)


# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=(data_characteristics["words_dim"], ), name='encoder_input', dtype="int32")
inputs_embeded = Embedding(len(embedding_matrix),
                            data_characteristics["embedding_dim"],
                            weights=[np.asarray(embedding_matrix)],
                            input_length=data_characteristics["words_dim"],
                            trainable=False)(inputs)
h = encoder_net_cnn(inputs_embeded, data_characteristics)
z_mean = Dense(latent_dim, name='z_mean', activation="sigmoid")(h)
z_log_var = Dense(latent_dim, name='z_log_var', activation="sigmoid")(h)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
decoder_outputs = decoder_net_cnn(latent_inputs, data_characteristics)

# instantiate decoder model
decoder = Model(latent_inputs, decoder_outputs, name='decoder')
decoder.summary()

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae')


# VAE loss = mse_loss or xent_loss + kl_loss
# reconstruction_loss = mse(inputs, outputs)
# reconstruction_loss *= (data_characteristics["words_dim"]*data_characteristics["embedding_dim"])
def inputs_onehot(inputs):
    i = inputs
    print("inputs", K.int_shape(i))
    i = K.one_hot(i, data_characteristics["vocabulary_dim"])
    print("inputs", K.int_shape(i))

    return i

reconstruction_loss = 0.5 * K.categorical_crossentropy(inputs_onehot(inputs), outputs)
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.sum(reconstruction_loss, axis=-1) + kl_loss
vae_loss = K.mean(vae_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer=Adam(lr=0.0003))

vae.summary()

vae_weights_dir = "weights/deconv_lvm_20ng_weights.h5y"
# if os.path.exists(vae_weights_dir):
#     vae.load_weights(vae_weights_dir)
#     print("load weights from", vae_weights_dir)


def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.

    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

class EpochCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if epoch>0:
            # save weights in disk
            vae.save_weights(vae_weights_dir)


            encoder_re = encoder.predict(X_test)
            print("z_mean", encoder_re[0])
            print("z_log_var", encoder_re[1])
            x_test_encoded = encoder_re[2]
            decoder_re = decoder.predict(x_test_encoded)

            loss_kl = 1 + encoder_re[1] - np.square(encoder_re[0]) - np.exp(encoder_re[1])
            print("kl loss", -0.5*np.mean(np.sum(loss_kl, axis=-1)))

            # X_test_onehot = to_categorical(X_test, data_characteristics["vocabulary_dim"])
            # loss_recon = 0
            # for i in range(len(X_test_onehot)):
            #     loss_recon_sample = 0
            #     sample_true = X_test_onehot[i]
            #     sample_pred = decoder_re[i]
            #     for j in range(len(sample_pred)):
            #         loss_recon_sample += metrics.log_loss(sample_true[j], sample_pred[j])
            #     loss_recon += loss_recon_sample
            # print("recon loss", 0.5* loss_recon/len(X_test_onehot))

            x_test_encoded_2d = PCA(n_components=2).fit_transform(x_test_encoded)
            plot_2d(x_test_encoded_2d, Y_test, epoch, "output/")
            y_test_pred = GaussianMixture(n_components=data_characteristics["num_classes"]).fit_predict(x_test_encoded)
            # y_train_pred = KMeans(n_clusters=data_characteristics["num_classes"], init='k-means++', max_iter=100, n_init=20,
            #     verbose=0).fit(x_train_encoded).labels_
            # print("Y_test", Y_test)
            # print("y_test_pred", y_test_pred)
            H, C, VM = metrics.homogeneity_completeness_v_measure(Y_test, y_test_pred)

            y = np.array(Y_test)
            y_pred = np.array(y_test_pred)
            D = max(y_pred.max(), y.max()) + 1
            w = np.zeros((D, D), dtype=np.int64)
            for i in range(y_pred.size):
                w[y_pred[i], y[i]] += 1
            ind = linear_assignment(w.max() - w)
            ACC = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

            print(" Epoch", epoch + 1, "ACC", ACC, "H", H, "C", C, "VM", VM)
epoch_callback = EpochCallback()


# fit model
vae.fit_generator(
    generate_batch_train_indexes_mem_x(train_set, batch_size, data_characteristics["words_dim"], word_index_dict),
    steps_per_epoch= data_characteristics["all_data_count"] // batch_size,
    epochs=epochs,
    verbose=1,
    callbacks=[epoch_callback],
    validation_data=None,
    validation_steps=None,
    class_weight=None,
    max_queue_size=3,
    workers=1,
    use_multiprocessing=True,
    shuffle=True,
    initial_epoch=0)

