#!/usr/bin/env Basics 3.6.1
# -*- coding: utf-8 -*-
# @Time    : 2018/5/11 13:41
# @Author  : Miao Li
# @File    : hello.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # ignore the log info at the beginning of run info

import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
intermed = tf.add(input2, input3)
mul = tf.multiply(input1, intermed)

with tf.Session():
  result = sess.run([input1, intermed])
  print(result)
