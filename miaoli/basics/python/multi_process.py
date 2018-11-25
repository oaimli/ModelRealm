# python 3.5
# -*- coding: utf-8 -*-
# @Time    : 2018/8/24 9:38
# @Author  : Miao Li
# @File    : prepare_data.py

"""
multiprocess template
"""
import argparse
import xml.etree.ElementTree as ET
import os
import pickle as pkl
import multiprocessing
from tqdm import tqdm
import functools
import json
import numpy as np

label_list_dir = ""

def get_graph_zj_multi_process(i, label_list, word_dict, G_image):
    item1 = label_list[i].split('\t')[1].strip()
    edge_list_zj = []
    len_label_list = len(label_list)
    for j in range(i + 1, len_label_list):
        item2 = label_list[j].split('\t')[1].strip()
        synsets1 = word_dict.get(item1, [])
        synsets2 = word_dict.get(item2, [])
        if len(synsets1) and len(synsets2):
            hops_mean = 0
            for word_i in synsets1:
                for word_j in synsets2:
                    if word_i in G_image.node and word_j in G_image.node:
                        # print(word_i, word_j)
                        hops = nx.dijkstra_path_length(G_image, source=word_i, target=word_j)
                        if hops < 4:
                            hops_mean += hops
            hops_mean = hops_mean / (len(synsets1) * len(synsets2))
            if hops_mean > 0:
                edge_list_zj.append((item1, item2, hops_mean))
    # print(i)
    return edge_list_zj

if __name__ == "__main__":
    with open(label_list_dir) as label_file:
        label_list = label_file.readlines()
        len_label_list = len(label_list)
        # parallel
        partial_get_graph_zj = functools.partial(get_graph_zj_multi_process, label_list=label_list, word_dict=[], G_image=[])
        with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
                edge_list_zj = list(tqdm(p.imap(partial_get_graph_zj, range(len_label_list), chunksize=16), total=len_label_list, desc="get_graph_zj"))

