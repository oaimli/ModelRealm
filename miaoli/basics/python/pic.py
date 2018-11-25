#!/usr/bin/env python 3.6.1
# -*- coding: utf-8 -*-
# @Time    : 2018/7/31 14:24
# @Author  : Miao Li
# @File    : pic.py

# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

#设置x,y轴的数值（y=sinx）
x = range(1, 21, 1)
y = [54.920, 73.670, 75.570, 78.350, 80.920, 79.87, 82.590, 84.030, 83.860, 83.850, 84.400, 83.710, 84.330, 83.090, 80.860, 84.060, 86.060, 86.120, 85.910,84.740]

#创建绘图对象，figsize参数可以指定绘图对象的宽度和高度，单位为英寸，一英寸=80px
plt.figure(figsize=(8,4))

#在当前绘图对象中画图（x轴,y轴,给所绘制的曲线的名字，画线颜色，画线宽度）
plt.plot(x,y,label="$sin(x)$",color="red",linewidth=2)

#X轴的文字
plt.xlabel("Epoch")

#Y轴的文字
plt.ylabel("Accuracy(%)")

#图表的标题
# plt.title("PyPlot First Example")

#Y轴的范围
plt.ylim(0,100)

#显示图示
# plt.legend()

plt.xticks(range(0, 22, 2))
#显示图
# plt.show()
#保存图
plt.savefig("sinx.jpg")
