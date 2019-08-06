#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2018/9/16 16:00
# @Author  : Miao Li
# @File    : get_embeddings.py

"""
load texts to the memory
"""
import pickle as pkl
import numpy as np
import os
import random
from tqdm import tqdm
import sys
import enchant

glove_path = "/home/lm/DataHDD/data/word_embeddings/glove.6B.300d.txt"
# glove_path = "/home/thy/limiao/data/glove.6B.300d.txt"
data_path = "../data"


def get_data_characteristics(dataset):
    # data characteristics
    data_characteristics = {}
    if dataset == "20ng":
        data_characteristics["words_dim"] = 125 #118
        data_characteristics["embedding_dim"] = 300
        data_characteristics["num_classes"] = 20
        # data_characteristics["all_data_count"] = 18846
    if dataset == "reuters-lyrl2004":
        data_characteristics["words_dim"] = 87 #119
        data_characteristics["embedding_dim"] = 300
        data_characteristics["num_classes"] = 4
        # data_characteristics["all_data_count"] = 685071
    if dataset == "GoogleNews-TS":
        data_characteristics["words_dim"] = 25
        data_characteristics["embedding_dim"] = 300
        data_characteristics["num_classes"] = 152
        # data_characteristics["all_data_count"] = 11109

    return data_characteristics


def load_embedding_glove(vocabulary, embedding_dim):
    """
    load glove word embedding
    :return:
    """
    # s = nltk.stem.snowball.EnglishStemmer()
    glove_embedding = {}
    with open(glove_path, "r") as f:
        for line in tqdm(f, desc="Get embedding from glove"):
            line = line.strip().split()
            glove_embedding[line[0]] = line[1:]
    # print(glove_embedding["sports"], glove_embedding["beaches"])

    words_embedding_dict = {}
    error_word_count = 0
    for word in tqdm(vocabulary, desc="Random embedding of words not in glove"):
        stem = word
        # if stem not in glove_embedding:
        #     # print(stem, vocabulary[stem])
        #     error_word_count += 1
        #     if vocabulary[stem]>10:
        #         words_embedding_dict[word] = np.random.randn(embedding_dim)
        if stem in glove_embedding:
            words_embedding_dict[word] = glove_embedding[stem]

    print("glove all", len(glove_embedding), ", words embedding", len(words_embedding_dict), ", no embedding", error_word_count)
    return words_embedding_dict


def filter_vocabulary(dataset):
    with open(os.path.join(data_path, dataset + ".voca.pkl"), "rb") as f:
        vocabulary = pkl.load(f)
    print("Vocabulary origin len", len(vocabulary))

    en_dict = enchant.Dict("en_US")

    vocabulary_new = {}
    for item in vocabulary:
        # delete stop words or filter words present too less times here
        word_len = len(list(item))
        if 2<=word_len and en_dict.check(item) and vocabulary[item]>25:
            vocabulary_new[item] = vocabulary[item]
    print("Vocabulary valid len", len(vocabulary_new))
    return vocabulary_new


def load_data(dataset, split_rate=0.1, split_count=1000):
    """
    load the whole texts in the memory, and divide it into testing and training set
    """
    vocabulary = filter_vocabulary(dataset)
    data_characteristics = get_data_characteristics(dataset)

    texts_path = os.path.join(data_path, dataset + ".txt")

    texts_labels = []
    with open(texts_path) as text_f:
        for line in tqdm(text_f):
            texts_labels.append(line.strip())
    print("all data", len(texts_labels))

    if split_count>0:
        test_sample_count = split_count
    else:
        test_sample_count = int(len(texts_labels) * split_rate)

    test_sample_index_list = set(random.sample([n for n in range(len(texts_labels))], test_sample_count))


    texts_labels_train = []
    texts_labels_test = []
    len_texts_labels = len(texts_labels)
    for  i in tqdm(range(len_texts_labels)):
        # delete words not in vocabulary
        words = texts_labels[i].split()
        words_new = []
        for word in words[:-1]:
            if word in vocabulary:
                words_new.append(word)
        text = " ".join(words_new) + " " + words[-1] # including label
        if i in test_sample_index_list:
            texts_labels_test.append(text)
        else:
            texts_labels_train.append(text)
    data_characteristics["all_data_count"] = len(texts_labels)

    del texts_labels

    word_embedding_dict = load_embedding_glove(vocabulary, data_characteristics["embedding_dim"])# not all words in the vocabulary are in words_embedding_dict
    print("Load data:","train %s"%len(texts_labels_train), "test %s"%len(texts_labels_test))

    word_index_dict = {}
    embedding_matrix = []
    index = 0
    for word in word_embedding_dict:
        word_index_dict[word] = index
        embedding_matrix.append(word_embedding_dict[word])
        index += 1

    embedding_matrix.append([0.]*data_characteristics["embedding_dim"])

    data_characteristics["vocabulary_dim"] = len(embedding_matrix)
    return texts_labels_train, texts_labels_test, word_embedding_dict, data_characteristics, word_index_dict, embedding_matrix


def statistic_length(dataset):
    words = filter_vocabulary(dataset)
    all = .0
    total_len = .0
    max_len = 0
    with open(os.path.join(data_path, dataset + ".txt")) as f:
        for line in f:
            all += 1
            curr_len = 0
            items = line.strip().split()[:-1]
            for word in items:
                if word in words:
                    total_len += 1
                    curr_len += 1
            if curr_len > max_len:
                max_len = curr_len
    avg_len = total_len / all

    print(sys._getframe().f_code.co_name, avg_len, max_len)


if __name__ == "__main__":
    statistic_length("GoogleNews-TS")
    