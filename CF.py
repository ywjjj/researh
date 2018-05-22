#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time
from scipy.stats import pearsonr
from MatrixFix import *
pd.options.mode.chained_assignment = None  # default='warn'
'''
pearson相关系数的平均为共同评分过的item的平均
预测中的平均为所有的平均
'''


# 基于pandas的标准协同过滤的实现
class CF(object):
    def __init__(self, data):
        self.data = data
        self.sim = {}
        self.sim_fix = {}

    # 原始pearson系数
    def pearson(self, user1, user2):
        user1 = int(user1)
        user2 = int(user2)
        if (user1, user2) in self.sim.keys():
            p = self.sim[(user1, user2)]
            return p
        user1_rating = self.data[self.data.userId == user1]
        itemlist = user1_rating.movieId.tolist()
        user2_rating = self.data[self.data.userId == user2]
        user2_same = user2_rating[user2_rating.movieId.isin(itemlist)]
        samelist = user2_same.movieId.tolist()
        user1_same = user1_rating[user1_rating.movieId.isin(samelist)]
        if len(samelist) == 0:
            p = -1
            self.sim[(user1, user2)] = p
            self.sim[(user2, user1)] = p
            return p
        x1 = user1_same['rating'].values
        x2 = user2_same['rating'].values
        p = pearsonr(x1, x2)[0]
        if np.isnan(p):
            p = -1
            self.sim[(user1, user2)] = p
            self.sim[(user2, user1)] = p
            return -1
        self.sim[(user1, user2)] = p
        self.sim[(user2, user1)] = p
        return p

    # 修正评分后的pearson系数
    def pearson_fix(self, user1, user2):
        user1 = int(user1)
        user2 = int(user2)
        if (user1, user2) in self.sim_fix.keys():
            p = self.sim_fix[(user1, user2)]
            return p
        user1_rating = self.data[self.data.userId == user1]
        itemlist = user1_rating.movieId.tolist()
        user2_rating = self.data[self.data.userId == user2]
        user2_same = user2_rating[user2_rating.movieId.isin(itemlist)]
        samelist = user2_same.movieId.tolist()
        user1_same = user1_rating[user1_rating.movieId.isin(samelist)]
        if len(samelist) == 0:
            p = -1
            self.sim_fix[(user1, user2)] = p
            self.sim_fix[(user2, user1)] = p
            return p
        x1 = user1_same['fix'].values
        x2 = user2_same['fix'].values
        p = pearsonr(x1, x2)[0]
        if np.isnan(p):
            p = -1
            self.sim_fix[(user1, user2)] = p
            self.sim_fix[(user2, user1)] = p
            return p
        self.sim_fix[(user1, user2)] = p
        self.sim_fix[(user2, user1)] = p
        return p

    def predict(self, user, item, k=0.6):
        user = int(user)
        item = int(item)
        user_rating = self.data[self.data.userId == user]
        aver = user_rating['rating'].mean()
        item_rating = self.data[self.data.movieId == item]
        item_rating['sim'] = item_rating['userId'].map(lambda x: self.pearson(user1=user, user2=x))
        item_rating = item_rating.dropna().sort_values('sim', ascending=False)
        if k > len(item_rating):
            k = len(item_rating)
        top_k = item_rating[item_rating.sim >= k]
        # top_k = item_rating.head(k)
        top_k['final'] = top_k['sim'] * (top_k['rating']-(top_k['userId'].map(lambda x: self.data[self.data.userId == x]['rating'].mean())))
        if top_k['sim'].sum() == 0:
            score = aver
        else:
            score = aver + (top_k['final'].sum() / top_k['sim'].sum())
        if score > 5:
            score = 5
        print(user, 'item', item, 'predict is', score)
        return score

    def predict_fix(self, user, item, k=0.6):
        user = int(user)
        item = int(item)
        user_rating = self.data[self.data.userId == user]
        aver = user_rating['rating'].mean()
        item_rating = self.data[self.data.movieId == item]
        item_rating['sim'] = item_rating['userId'].map(lambda x: self.pearson_fix(user1=user, user2=x))
        item_rating = item_rating.dropna().sort_values('sim', ascending=False)
        if k > len(item_rating):
            k = len(item_rating)
        top_k = item_rating[item_rating.sim >= k]
        top_k['final'] = top_k['sim'] * (top_k['rating']-(top_k['userId'].map(lambda x: self.data[self.data.userId == x]['rating'].mean())))
        if top_k['sim'].sum() == 0:
            score = aver
        else:
            score = aver + (top_k['final'].sum() / top_k['sim'].sum())
        if score > 5:
            score = 5
        return score

    def mae(self, test):
        test['predict'] = test.apply(lambda row: self.predict(row['userId'], row['movieId']), axis=1)
        test['diff'] = abs(test['rating'] - test['predict'])
        mae = test['diff'].sum()/len(test)
        return mae

    def mae_fix(self, test):
        test['predict'] = test.apply(lambda row: self.predict_fix(row['userId'], row['movieId']), axis=1)
        test['diff'] = abs(test['rating'] - test['predict'])
        mae = test['diff'].sum() / len(test)
        return mae

if __name__ == '__main__':
    start_time = time.time()
    data_types = {'userId': 'int32', 'movieId': 'int32',
                  'rating': 'float64', 'timestamp': 'int32'}
    ratings = pd.read_csv('E:/dataset/ml-1m/ratings.dat', sep='::', header=None,
                          names=['userId', 'movieId', 'rating', 'timestamp'], dtype=data_types)
    train, test = train_test_split(ratings, test_size=0.2, random_state=0)
    print('running bayes')
    bayes = BayesRatins(train)
    bayes_mean = bayes.get_bayesian_estimates()
    matrix = MatrixFix(train, bayes_mean)
    fixed_matrix = matrix.fix()
    print('running cf')
    cf = CF(fixed_matrix)
    print('标准协同过滤.......')
    mae = cf.mae(test)
    print('修正协同过滤.......')
    mae_fix = cf.mae_fix(test)
    print('标准协同过滤MAE是', mae)
    print('修正矩阵后协同过滤MAE是', mae_fix)
    end_time = time.time()
    print('running time', end_time - start_time)