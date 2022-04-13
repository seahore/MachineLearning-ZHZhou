#!/usr/bin/python3 
# -*- coding: utf-8 -*-

from time import time
import numpy as np
from sklearn import tree
import graphviz

data = [
    [ 0.697, 0.460 ],
    [ 0.774, 0.376 ],
    [ 0.634, 0.264 ],
    [ 0.608, 0.318 ],
    [ 0.556, 0.215 ],
    [ 0.403, 0.237 ],
    [ 0.481, 0.149 ],
    [ 0.437, 0.211 ],
    [ 0.666, 0.091 ],
    [ 0.243, 0.267 ],
    [ 0.245, 0.057 ],
    [ 0.343, 0.099 ],
    [ 0.639, 0.161 ],
    [ 0.657, 0.198 ],
    [ 0.360, 0.370 ],
    [ 0.593, 0.042 ],
    [ 0.719, 0.103 ],
]

target = [1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1]

class ensemble_decision_tree:
    def __init__(self):
        self.__learners : list[tree.DecisionTreeClassifier] = []
        self.__weights : list[float] = []
    def add(self, learner:tree.DecisionTreeClassifier, weights:float):
        self.__learners.append(learner)
        self.__weights.append(weights)
    def evaluate(self, test_data:list[float]) -> float:
        z = np.sum(self.__weights)
        for i in range(len(self.__weights)):
            self.__weights[i] /= z
        sum = 0.0
        for i in range(len(self.__learners)):
            sum += self.__weights[i] * self.__learners[i].predict([test_data])[0]
        return sum


def ada_boost(data:list[list[float]], labels:list[float], epoch:int) -> ensemble_decision_tree:
    dist = [1 / len(data)] * len(data)
    esm_tree = ensemble_decision_tree()
    rand = np.random.default_rng(int(time()))
    for t in range(epoch):
        # 训练基分类器，得出其分类结果
        dtree = tree.DecisionTreeClassifier()
        dtree.fit(rand.choice(data, len(data), p=dist), labels)



        pred = dtree.predict(data)
        # 估计误差
        err = 0.0
        for i in range(len(pred)):
            if pred[i] != labels[i]:
                err += 1 / len(pred)
        if err > 0.3:    # err > 0.5 抛弃的话，似乎不行
            continue
        if err < 1E-10:
            ultimate_tree = ensemble_decision_tree()
            ultimate_tree.add(dtree, 1.0)
            return ultimate_tree
        # 确定权重
        weight = 0.5 * np.log((1.0-err)/err)
        # 更新样本分布
        for i in range(len(pred)):
            dist[i] *= np.exp(weight if pred[i] != labels[i] else -weight)
        # 规范化
        z = np.sum(dist)
        for i in range(len(pred)):
            dist[i] /= z
        # 集成
        esm_tree.add(dtree, weight)
    return esm_tree





esm_tree = ada_boost(data, target, 100)
err = 0.0
for i in range(len(data)):
    print(f"Prediction {i}: {esm_tree.evaluate(data[i])}")


'''
        dot_data = tree.export_graphviz(dtree, out_file=None,
                     feature_names=["Density", "Sweetness"],
                     class_names=["Good", "Bad"],
                     filled=True, rounded=True,
                     special_characters=True,
                     fontname="Fira Sans")
        graph = graphviz.Source(dot_data)
        graph.render("suika")
'''