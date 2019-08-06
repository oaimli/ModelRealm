#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2018/9/16 16:00
# @Author  : Miao Li
# @File    : store_texts.py

"""
Store the origin data and its vocabulary in special format
"""
import scipy.io as scio
import json
import pickle as pkl
import numpy as np
import os
import sys
from collections import defaultdict
from tqdm import tqdm


origin_data_path = "/home/lm/DataHDD/data"


def store_texts_vocabulary(dataset, texts, vocabulary):
    texts_file = "../data/" + dataset + ".txt"
    with open(texts_file, "w") as f_texts:
        f_texts.writelines(texts)

    vocabulary_file = "../data/" + dataset + ".voca.pkl"
    with open(vocabulary_file, "wb") as f_voca:
        pkl.dump(vocabulary, f_voca)

    print("vocabulary", len(vocabulary))

    print(dataset, "stored")


def get_20ng_GSDPMM():
    """
    pre process 20ng and store 20ng in special format
    :return:
    """
    dataset = "20ng"
    dataset_path = "../../GSDMM-master/data/20ng"

    # store the texts and the vocabulary
    texts = []
    vocabulary = {}
    with open(dataset_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            doc = data["text"]
            label = data["cluster"]
            words = doc.strip().split()
            for word in words:
                count = vocabulary.get(word, 0)
                vocabulary[word] = count + 1
            texts.append(doc.strip() + " " + str(label) + "\n")

    store_texts_vocabulary(dataset, texts, vocabulary)


def get_GoogleNews_TS_GSDPMM():
    """
    pre process Google News TS and store it in special format
    :return:
    """
    dataset = "GoogleNews-TS"
    dataset_path = "../../GSDMM-master/data/TS"

    # store the texts and the vocabulary
    texts = []
    vocabulary = {}
    labels = set([])
    with open(dataset_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            doc = data["text"]
            label = data["cluster"]
            labels.add(label)
            words = doc.strip().split()
            for word in words:
                count = vocabulary.get(word, 0)
                vocabulary[word] = count + 1
            texts.append(doc.strip() + " " + str(label) + "\n")

    print(dataset, "all count", len(texts), "num classes", len(labels))
    store_texts_vocabulary(dataset, texts, vocabulary)


def get_lyrl2004():
    """
    get reuters-lyrl2004 document and store it
    :return:
    """
    dataset = "reuters-lyrl2004"
    dataset_path = os.path.join(origin_data_path, dataset)

    id_category_dict = defaultdict(set)
    with open(os.path.join(dataset_path, "rcv1-v2.topics.qrels")) as f:
        for line in f:
            line = line.strip().split()
            id_category_dict[line[1]].add(list(line[0])[0])

    invalid_ids = []
    id_label_dict ={}
    for id in id_category_dict:
        categories = id_category_dict[id]
        if len(categories) > 1:
            invalid_ids.append(id)
        else:
            if "C" in categories:
                id_label_dict[id] = 0
            if "E" in categories:
                id_label_dict[id] = 1
            if "G" in categories:
                id_label_dict[id] = 2
            if "M" in categories:
                id_label_dict[id] = 3

    print("invalid", len(invalid_ids))


    files = os.listdir(dataset_path)

    texts = []
    vocabulary = {}
    all = 0
    valid = 0
    for f in files:
        if f.startswith("lyrl2004"):
            with open(os.path.join(dataset_path, f), 'r') as f:
                document = []
                for line in f:
                    if line.strip() == "":
                        all += 1
                        # process current document
                        id = document[0].split()[1]
                        if id not in invalid_ids:
                            valid += 1
                            label = str(id_label_dict[id])
                            x = " ".join(document[2:])
                            for word in x.strip().split():
                                vocabulary[word] = vocabulary.get(word, 0) + 1
                            texts.append(x.strip() + " " + str(label) + "\n")
                        document = []
                    else:
                        document.append(line.strip())

    print("all", all, "valid", valid)
    store_texts_vocabulary(dataset, texts, vocabulary)

def get_yahoo_answers():
    dataset = "yahoo_answers_csv"
    dataset_path = os.path.join(origin_data_path, dataset)

if __name__ == "__main__":
    # get_20ng_GSDPMM()
    # get_lyrl2004()
    # get_GoogleNews_TS_GSDPMM()
    get_yahoo_answers()