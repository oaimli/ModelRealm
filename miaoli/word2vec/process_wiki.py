#!/usr/bin/env Basics 3.6.1
# -*- coding: utf-8 -*-
# @Time    : 2018/9/11 20:48
# @Author  : Miao Li
# @File    : process_wiki.py

"""
将wiki从XML格式转化为text格式
"""


if __name__ == '__main__':
    # 抽取text
    # from gensim.corpora import WikiCorpus
    # inp = "E:\Word2Vec\data\zhwiki\zhwiki-latest-pages-articles.xml.bz2"
    # outp = "E:\Word2Vec\data\zhwiki\wiki.zn.text"
    #
    # i = 0
    # output = open(outp, 'w', encoding="utf-8")
    # wiki = WikiCorpus(inp, lemmatize=False, dictionary={})
    # for text in wiki.get_texts():
    #     output.write(" ".join(text) + "\n")
    #     i = i + 1
    #     if (i % 10000 == 0):
    #         print("Saved " + str(i) + " articles")
    #
    # output.close()
    # print("Finished Saved " + str(i) + " articles")


    # 简繁转换
    # f_in = open("E:\Word2Vec\data\zhwiki\wiki.zn.text",encoding="utf-8")
    # f_out = open("E:\Word2Vec\data\zhwiki\wiki.zn.s.text", "w", encoding="utf-8")
    # from opencc import OpenCC
    # openCC = OpenCC('t2s')
    # tmp = []
    # for x in range(314510):
    #     tmp.append(openCC.convert(f_in.readline()) + "\n")
    #     if x > 0 and x % 10000 == 0:
    #         print(x)
    #         f_out.writelines(tmp)
    #         tmp = []
    # f_out.writelines(tmp)
    # tmp = []


    # # 分词
    # f_in = open("E:\Word2Vec\data\zhwiki\wiki.zn.s.text", encoding="utf-8")
    # f_out = open("E:\Word2Vec\data\zhwiki\wiki.zn.s.word.text", "w", encoding="utf-8")
    # import jieba
    # lines = f_in.readlines()
    # print(len(lines) / 2)
    # tmp = []
    # x = 0
    # for line in lines:
    #     line = line.strip()
    #     if line != "":
    #         x += 1
    #         tmp.append("/".join(jieba.cut(line, cut_all=False, HMM=True)) + "\n")
    #         if x > 0 and x % 10000 == 0:
    #             print(x)
    #             f_out.writelines(tmp)
    #             tmp = []
    #
    # f_out.writelines(tmp)
    # tmp = []


    # # 分词后去掉空格，统计词数量
    # f_in = open("E:\Word2Vec\data\zhwiki\wiki.zn.s.word.text", encoding="utf-8")
    # lines = f_in.readlines()
    # word_map = {}
    # for line in lines:
    #     words = line.strip().split("/")
    #     words_new = []
    #     for word in words:
    #         if word.strip() != "":
    #             word_map[word] = word_map.get(word, 0) + 1
    #
    # print(len(word_map))
    #
    # f_out = open("E:\Word2Vec\data\zhwiki\wiki.zn.s.entry.frequency.text", "w", encoding="utf-8")
    # items = []
    # x = 0
    # for item in word_map:
    #     x += 1
    #     items.append(item + " " + str(word_map[item]) + "\n")
    #     if x % 10000 == 0:
    #         f_out.writelines(items)
    #         items = []
    # f_out.writelines(items)


    f_in = open("E:\word2vec\data\zhwiki\wiki.zn.s.entry.frequency.text", encoding="utf-8")
    freq = 2
    f_out = open("E:\word2vec\data\zhwiki\wiki.zn.s.entry.frequency." + str(freq) + ".text", "w", encoding="utf-8")
    lines = f_in.readlines()
    all = 0
    out = []
    for line in lines:
        item = line.strip().split(" ")
        all += int(item[1])
        if int(item[1]) <=freq:
            out.append(line)
    print(all)
    print(len(out))
    f_out.writelines(out)