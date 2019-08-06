#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2018/9/16 16:00
# @Author  : Miao Li
# @File    : get_embeddings.py

"""
get embeddings from the texts in the memory
"""
import numpy as np
import random


def get_doc_embedding(doc, words_dim, embedding_dim, word_embedding_dict):
    """
    get embedding of each doc, padding or trimming
    :param doc:
    :param words_dim:
    :param embedding_dim:
    :param word_embedding_dict:
    :return:
    """
    items = doc.strip().split()[:-1]
    embedding = []
    for word in items:
        if word in word_embedding_dict:
            embedding.extend(word_embedding_dict[word])

    if len(embedding)<words_dim*embedding_dim:
        embedding.extend([.0]*(words_dim*embedding_dim-len(embedding)))
    else:
        embedding = embedding[:words_dim*embedding_dim]

    if len(embedding) != words_dim*embedding_dim:
        print("Embedding len error!")

    return embedding


def get_doc_indexes(doc, words_dim, word_index_dict, vocabulary_dim):
    items = doc.strip().split()[:-1]
    doc_indexes = []
    for word in items:
        if word in word_index_dict:
            doc_indexes.append(word_index_dict[word])

    if len(doc_indexes) < words_dim:
        doc_indexes.extend([vocabulary_dim]*(words_dim-len(doc_indexes)))
    else:
        doc_indexes = doc_indexes[:words_dim]

    if len(doc_indexes) != words_dim:
        print("Embedding len error!")

    return doc_indexes


def generate_batch_train_data_mem_cla(texts_labels_train, batch_size, words_dim, embedding_dim, word_embedding_dict, num_classes):
    while True:
        batch_texts_embedding = []
        batch_labels = []
        for text in texts_labels_train:
            batch_texts_embedding.append(get_doc_embedding(text, words_dim, embedding_dim, word_embedding_dict))
            batch_labels.append(int(text.split()[-1]))

            if len(batch_texts_embedding) == batch_size:
                # print("count", count)
                batch_texts_embedding = np.array(batch_texts_embedding)
                batch_labels = to_categorical(batch_labels, num_classes)
                yield (np.array(batch_texts_embedding), np.array(batch_labels))
                batch_texts_embedding =[]
                batch_labels = []


def get_test_embedding_mem_cla(texts_labels_test, words_dim, embedding_dim, word_embedding_dict, num_classes):
    texts_embedding_test = []
    labels_test = []
    for item in texts_labels_test:
        texts_embedding_test.append(get_doc_embedding(item, words_dim, embedding_dim, word_embedding_dict))
        labels_test.append(int(item.split()[-1]))
    labels_test = to_categorical(labels_test, num_classes)
    return np.array(texts_embedding_test), np.array(labels_test)


# for VaDE
def generate_batch_train_data_mem_vae_xy(texts_labels_train, batch_size, words_dim, embedding_dim, word_embedding_dict, num_classes):
    while True:
        batch_texts_embedding = []
        batch_labels = []
        for text in texts_labels_train:
            batch_texts_embedding.append(get_doc_embedding(text, words_dim, embedding_dim, word_embedding_dict))
            batch_labels.append(int(text.split()[-1]))

            if len(batch_texts_embedding) == batch_size:
                # print("count", count)
                batch_texts_embedding = np.array(batch_texts_embedding)
                # batch_labels = to_categorical(batch_labels, num_classes)
                yield (np.array(batch_texts_embedding), np.array(batch_texts_embedding))
                batch_texts_embedding =[]
                batch_labels = []


# for nvtc and representation learning
def generate_batch_train_data_mem_vae_x(texts_labels_train, batch_size, words_dim, embedding_dim, word_embedding_dict, num_classes):
    while True:
        batch_texts_embedding = []
        batch_labels = []
        for text in texts_labels_train:
            batch_texts_embedding.append(get_doc_embedding(text, words_dim, embedding_dim, word_embedding_dict))
            batch_labels.append(int(text.split()[-1]))

            if len(batch_texts_embedding) == batch_size:
                # print("count", count)
                batch_texts_embedding = np.array(batch_texts_embedding)
                # batch_labels = to_categorical(batch_labels, num_classes)
                yield (np.array(batch_texts_embedding), None)
                batch_texts_embedding =[]
                batch_labels = []


def get_test_embedding_mem_vae(texts_labels_test, words_dim, embedding_dim, word_embedding_dict, num_classes):
    texts_embedding_test = []
    labels_test = []
    for item in texts_labels_test:
        texts_embedding_test.append(get_doc_embedding(item, words_dim, embedding_dim, word_embedding_dict))
        labels_test.append(int(item.split()[-1]))
    # labels_test = to_categorical(labels_test, num_classes)
    return np.array(texts_embedding_test), np.array(labels_test)





def generate_batch_train_indexes_mem_x(texts_labels_train, batch_size, words_dim, word_index_dict):
    vocabulary_dim = len(word_index_dict.keys())
    while True:
        batch_texts_indexes = []
        # batch_texts_onehot = []
        batch_labels = []
        for text in texts_labels_train:
            doc_indexes = get_doc_indexes(text, words_dim, word_index_dict, vocabulary_dim)
            batch_texts_indexes.append(doc_indexes)
            # batch_texts_onehot.append(to_categorical(doc_indexes, vocabulary_dim+1))
            batch_labels.append(int(text.split()[-1]))

            if len(batch_texts_indexes) == batch_size:
                # print("count", count)
                # batch_texts_onehot = np.array(batch_texts_onehot)
                batch_texts_indexes = np.array(batch_texts_indexes)
                yield (batch_texts_indexes, None)
                # batch_texts_onehot =[]
                batch_labels = []
                batch_texts_indexes = []


def get_test_indexes_mem_vae(texts_labels_test, words_dim, word_index_dict):
    vocabulary_dim = len(word_index_dict.keys())
    texts_onehot_test = []
    labels_test = []
    for item in texts_labels_test:
        texts_onehot_test.append(get_doc_indexes(item, words_dim, word_index_dict, vocabulary_dim))
        labels_test.append(int(item.split()[-1]))
    return np.array(texts_onehot_test), np.array(labels_test)


def generate_batch_train_indexes_mem_cla(texts_labels_train, batch_size, words_dim, word_index_dict, num_classes):
    vocabulary_dim = len(word_index_dict.keys())
    while True:
        batch_texts_indexes = []
        batch_labels = []
        for text in texts_labels_train:
            doc_indexes = get_doc_indexes(text, words_dim, word_index_dict, vocabulary_dim)
            batch_texts_indexes.append(doc_indexes)
            batch_labels.append(int(text.split()[-1]))

            if len(batch_texts_indexes) == batch_size:
                # print("count", count)
                batch_texts_indexes = np.array(batch_texts_indexes)
                batch_labels = to_categorical(batch_labels, num_classes)
                yield (batch_texts_indexes, np.array(batch_labels))
                batch_labels = []
                batch_texts_indexes = []


def get_test_indexes_mem_cla(texts_labels_test, words_dim, word_index_dict, num_classes):
    vocabulary_dim = len(word_index_dict.keys())
    texts_onehot_test = []
    labels_test = []
    for item in texts_labels_test:
        texts_onehot_test.append(get_doc_indexes(item, words_dim, word_index_dict, vocabulary_dim))
        labels_test.append(int(item.split()[-1]))
    labels_test = to_categorical(labels_test, num_classes)
    return np.array(texts_onehot_test), np.array(labels_test)



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