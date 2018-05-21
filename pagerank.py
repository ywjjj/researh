#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import networkx as nx
import pickle
import csv

if __name__ == '__main__':
    # 读取文件
    df = pd.read_csv('E:/dataset/C/docidrid.dat')
    # 构建pagerank
    G = nx.from_pandas_edgelist(df, 'docid', 'rid')
    pr = nx.pagerank_numpy(G, alpha=0.85)
    # 输出模型
    # with open('pr.pkl', 'wb') as f:
    #     pickle.dump(pr, f)
    # 载入模型
    # with open('pr.pkl', 'rb') as f:
    #     pr = pickle.load(f)
    # # 写入csv
    # with open('pagerank.csv', 'w') as f:
    #     csvWriter = csv.writer(f)
    #     for k, v in pr.items():
    #         csvWriter.writerow([k, v])
