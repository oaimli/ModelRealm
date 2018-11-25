# python 3.5
# -*- coding: utf-8 -*-
# @Time    : 2018/8/19 10:29
# @Author  : Miao Li
# @File    : vae_mnist.py

import os
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tqdm


def get_file_list(folder, file_list):
    if os.path.isfile(folder):
        file_list.append(folder)
    elif os.path.isdir(folder):
        for s in os.listdir(folder):
            new_dir = os.path.join(folder, s)
            get_file_list(new_dir, file_list)
    return file_list


# only run on GPU of index 0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# read data
images = get_file_list("./data/tchi/DatasetA_train_20180813/train_labeled/", [])
zj_data = []
count = 0
for image in images:
    im = Image.open(image)
    im = im.convert('RGB')
    iml = np.array(im).tolist()
    zj_data.append(iml)
    # if count == 10000:
    #     break
    # count += 1
zj_data = np.array(zj_data)
print(zj_data.shape)


tf.reset_default_graph()

batch_size = 64

X_in = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 3], name='X')
Y = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 3], name='Y')
Y_flat = tf.reshape(Y, shape=[-1, 64 * 64 * 3])
keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')

n_latent = 8


def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))


def encoder(X_in, keep_prob):
    activation = lrelu
    with tf.variable_scope("encoder", reuse=None):
        x = tf.layers.conv2d(X_in, filters=32, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=32, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=16, kernel_size=4, strides=1, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.contrib.layers.flatten(x)
        mn = tf.layers.dense(x, units=n_latent)
        sd = 0.5 * tf.layers.dense(x, units=n_latent)
        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_latent]))
        z = mn + tf.multiply(epsilon, tf.exp(sd))
        return z, mn, sd


def decoder(sampled_z, keep_prob):
    with tf.variable_scope("decoder", reuse=None):
        x = tf.layers.dense(sampled_z, units=4*4*3, activation=lrelu)
        x = tf.reshape(x, [-1, 4, 4, 3])
        x = tf.layers.conv2d_transpose(x, filters=16, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=32, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=32, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)

        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=64 * 64 * 3, activation=tf.nn.sigmoid)
        img = tf.reshape(x, shape=[-1, 64, 64, 3])
        return img


sampled, mn, sd = encoder(X_in, keep_prob)
dec = decoder(sampled, keep_prob)

unreshaped = tf.reshape(dec, [-1, 64*64*3])

img_loss = tf.reduce_sum(tf.squared_difference(unreshaped, Y_flat), 1)
latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
loss = tf.reduce_mean(img_loss + latent_loss)
optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)


sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(25):
    print("epoch: " + str(epoch))
    for i in range(zj_data.shape[0]):
        batch = zj_data[i: i+batch_size]
        sess.run(optimizer, feed_dict = {X_in: batch, Y: batch, keep_prob: 0.8})
        if not i % 3000:
            ls, d, i_ls, d_ls, mu, sigm = sess.run([loss, dec, img_loss, latent_loss, mn, sd], feed_dict={X_in: batch, Y: batch, keep_prob: 1.0})
            plt.imshow(np.reshape(batch[0], [64, 64, 3]))
            plt.axis('off')
            plt.show()
            plt.imshow(d[0])
            plt.axis('off')
            plt.show()
            print("Epoch: " + str(epoch) + "/25", "Index: " + str(i) + "/" + str(zj_data.shape[0]), "Loss: " + str(ls), "Image loss:" + str(np.mean(i_ls)), "Latent loss: " + str(np.mean(d_ls)))
        i += batch_size


    # generate image
    randoms = [np.random.normal(0, 1, n_latent) for _ in range(10)]
    imgs = sess.run(dec, feed_dict={sampled: randoms, keep_prob: 1.0})
    imgs = [np.reshape(imgs[i], [64, 64, 3]) for i in range(len(imgs))]
    for img in imgs:
        plt.figure(figsize=(1, 1))
        plt.axis('off')
        plt.imshow(img)


