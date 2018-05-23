#!/user/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from itertools import islice
from sklearn.model_selection import train_test_split
# np.set_printoptions(threshold=np.inf)

class Node(object):
    def __init__(self):
        self.neighbours = []


class ItemRank(object):
    def __init__(self, np_data):
        self.movie_names = []
        self.user_names = []
        self.movie_nodes = {}
        self.user_nodes = {}
        self.data = np_data

    # 生成图模型
    def generate_graph(self):
        # node = Node()
        # print("******生成图模型中......")
        self.movie_names = list(set(self.data[:, 1].astype(int)))
        # self.movie_names = set(self.data.movieId.tolist())
        print(self.movie_names)
        self.user_names = list(set(self.data[:, 0].astype(int)))
        # print(self.user_names)
        # self.user_names = set(self.data.userId.tolist())
        self.movie_nodes = {}
        self.user_nodes = {}
        for movie in self.movie_names:
            node = Node()
            node.name = movie
            self.movie_nodes[movie] = node
        for user in self.user_names:
            node = Node()
            node.name = user
            self.user_nodes[user] = node
        # 如果用户看过某部电影，则将这部电影加入到用户的neighbours中；对电影同样如此
        for i in range(len(self.data[:, 0])):
            self.user_nodes[self.data[i, 0].astype(int)].neighbours.append(self.movie_nodes[self.data[i, 1].astype(int)])
            self.movie_nodes[self.data[i, 1].astype(int)].neighbours.append(self.user_nodes[self.data[i, 0].astype(int)])

    # 根据图模型生成相关系数矩阵
    def generate_coef_from_graph(self):
        print("******此刻正在计算相关系数矩阵......")
        correlation_matrix = np.zeros((len(self.movie_names), len(self.movie_names)))
        for movie_name in self.movie_nodes.keys():
            for user in self.movie_nodes[movie_name].neighbours:
                for movie in user.neighbours:
                    if movie != self.movie_nodes[movie_name]:
                        correlation_matrix[self.movie_names.index(movie_name), self.movie_names.index(movie.name)] += 1
        for c in range(len(correlation_matrix[1, :])):
            correlation_matrix[:, c] /= sum(correlation_matrix[:, c])
        self.correlation_matrix = correlation_matrix

    # itemrank公式
    def item_rank(self, alpha, ir, d):
        print("******计算itemrank中......")
        return alpha * np.dot(self.correlation_matrix, ir) + (1 - alpha) * d

    # 生成评分向量
    def generate_d(self, user_name):
        print("******生成评分向量中中......")
        d = np.zeros(len(self.movie_names))
        for i in range(len(self.data[:, 0])):
            if self.data[i, 0].astype(int) == user_name:
                d[self.movie_names.index(self.data[i, 1].astype(int))] = self.data[i, 2].astype(float)
        return d


if __name__ == "__main__":
    # with open("/Users/JiaoFusen/Desktop/ml-latest-small/ratings.csv") as file:
    #     data = []
    #     for line in islice(file, 1, None):
    #         data.extend(line.rstrip("\n").split(","))
    # np_data = np.array(data).reshape(-1, 4)
    pd_data = pd.read_csv("/Users/JiaoFusen/Desktop/ml-latest-small/ratings.csv")
    # print(pd_data)
    np_data = pd_data.values
    print(np_data)
    # train_data, test_data = train_test_split(np_data, train_size=0.8)
    train_data = np_data
    item_rank = ItemRank(train_data)
    item_rank.generate_graph()
    # item_rank.generate_coef_from_graph()
    # 选取405号用户来进行计算
    # d = item_rank.generate_d(user_name=665)
    # IR = np.ones(len(item_rank.movie_names))
    # IR = d
    # 迭代计算
    # covered = False
    # counter = 0
    # while not covered:
    #     counter += 1
    #     old_IR = IR
    #     IR = item_rank.item_rank(0.85, IR, d)
    #     covered = (old_IR - IR < 0.0001).all()
    # print("after", counter, "counts")
    # print("IR now is ", IR.shape)
