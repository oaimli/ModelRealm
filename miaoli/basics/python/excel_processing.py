#!/usr/bin/python3
# -*- coding: utf-8 -*-

from tqdm import tqdm
import xlrd
import csv


book = xlrd.open_workbook("data.xlsx")
table_all = book.sheet_by_name(u'对照组全部数据')
table_1  = book.sheet_by_name(u'实验组1id')
table_2 = book.sheet_by_name(u'实验组2id')
table_3 = book.sheet_by_name(u'实验组3id')
sheets = [table_1, table_2, table_3]


all_value = {}
for i in tqdm(range(table_all.nrows)):
        value = table_all.row_values(i)
        all_value[str(value[0])] = value
print(len(all_value))


for sheet_index in range(len(sheets)):
    table = sheets[sheet_index]
    data = []
    col0 = table.col_values(0)
    print(len(col0))
    for i in tqdm(range(table.nrows)):
        v = str(col0[i])
        if v in all_value:
            value = all_value[v]
            data.append(value)
        else:
            data.append([v])
    with open("id%s.csv"%sheet_index, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(data)

