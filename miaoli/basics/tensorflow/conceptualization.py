#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2018/9/19 11:07
# @Author  : Miao Li
# @File    : conceptualization.py

"""Module summary.

Module description.
   
Module usage.
"""

d = "seahawks targeting antoine winfield updating previous report seattle seahawks sign free agent cb antoine winfield seahawks help replace cbs brandon browner walter thurmond facing league suspension source nfl gregg"

import urllib.request as uq

for word in d.split():
    fhand = uq.urlopen("https://concept.research.microsoft.com/api/Concept/ScoreByProb?instance=%s&topK=10" % word)
    data = fhand.read()
    print('data---\n',data)