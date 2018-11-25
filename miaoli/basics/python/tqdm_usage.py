# python 3.5
# -*- coding: utf-8 -*-
# @Time    : 2018/8/21 15:13
# @Author  : Miao Li
# @File    : tqdm_usage.py

import time
from tqdm import *
for i in tqdm(range(1000)):
       time.sleep(0.01)
