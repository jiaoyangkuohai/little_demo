# -*- coding:utf-8 -*-
"""
@author    : shihongliang
@time      : 2022-03-20 20:15
@filename  : adaboostDemo.py
"""

import numpy as np
import pandas as pd
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_rows", 20)
pd.set_option("display.width", 2000)
pd.set_option("display.max_colwidth", 2000)


class G:
    def __init__(self, threshold, which_type="<="):
        self.threshold = threshold
        self.which_type = which_type

    def __call__(self, x):
        if self.which_type == "<=":
            a = (x <= self.threshold) * 1
        elif self.which_type == ">=":
            a = (x >= self.threshold) * 1
        else:
            raise TypeError("类型错误, 必须是<= 或 >=")
        b = (a - 1) + a
        return b


class AdaBoostClassifier:
    def __init__(self, begin_w, x, y, m, threshold_candidate):
        self.m = m
        self.threshold_candidate = threshold_candidate
        self.D_list = []
        self.F_list = []
        self.begin_w = begin_w
        self.D_list.append(begin_w)
        self.x = x
        self.y = y

    def get_threshold_err(self, w):
        all = []
        for i in self.threshold_candidate:
            g_i_0 = G(i, "<=")
            g_i_1 = G(i, ">=")
            e_0 = np.sum((g_i_0(self.x) != self.y) * w)
            e_1 = np.sum((g_i_1(self.x) != self.y) * w)
            all.append((f"{i}_<=", e_0, g_i_0))
            all.append((f"{i}_>=", e_1, g_i_1))
        all = sorted(all, key=lambda x: x[1])
        return all[0]

    def get_alpha(self, e):
        return 0.5 * np.log((1 - e) / e)

    def get_next_weight(self, w, alpha, x, y, G_m):
        tmp = np.exp(-alpha * y * G_m(x)) * w
        return tmp / np.sum(tmp)

    def one_step(self, w):
        threshold_num, e, Gm = self.get_threshold_err(w)
        alpha = self.get_alpha(e)
        self.F_list.append((alpha, Gm, e))
        return self.get_next_weight(w, alpha, self.x, self.y, Gm)

    def fit(self):
        for i in range(self.m):
            w = self.one_step(self.D_list[i])
            self.D_list.append(w)
        return self.summary()

    def summary(self):
        df = pd.DataFrame({"w": [np.around(wi, 5) for wi in self.D_list],
                           "alpha": [g[0] for g in self.F_list] + ["-"],
                           "threshold": [g[1].threshold for g in self.F_list] + ["-"],
                           "type": [g[1].which_type for g in self.F_list]+["-"],
                           "err": [g[2] for g in self.F_list] + ["-"]})
        return df

    def predict(self, x):
        out = np.zeros_like(x, dtype=np.float32)
        for alpha, Gm, _ in self.F_list:
            out += alpha * Gm(x)
        return 2 * (out > 0) - 1


def main():
    # x 值
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32)
    # y 值
    y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1], dtype='i4')
    # 初始权重
    D1 = np.ones(10) * 0.1
    # 次数M
    m = 4
    # 阈值搜索空间
    threshold_candidate = np.array(range(1, 21, 2)) * 0.5
    # AdaBoost
    model = AdaBoostClassifier(D1, x, y, m, threshold_candidate)
    # 训练
    model.fit()
    # 查看训练结果
    print(model.summary())
    # 对x进行预测
    predict = model.predict(x)
    print(predict)
    # 将真实结果与预测结果对比
    print(predict == y)


if __name__ == '__main__':
    main()


