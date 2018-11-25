# python 3.5
# -*- coding: utf-8 -*-
# @Time    : 2018/8/21 18:59
# @Author  : Miao Li
# @File    : softmax_regression.py

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

mnist = input_data.read_data_sets('./ZSL-TChi/data/MNIST', one_hot=True)
print(mnist.train.images.shape)
print(mnist.train.labels.shape)

x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
# loss function
y_ = tf.placeholder(dtype=tf.float32, shape=[None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(mnist.train.images.shape[0]):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))