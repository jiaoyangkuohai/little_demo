# -*- coding:utf-8 -*-
"""
@author    : shihongliang
@time      : 2022-03-20 20:15
@filename  : adaboostDemo.py
"""
from typing import List

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", 20)
pd.set_option("display.max_rows", 20)
pd.set_option("display.width", 2000)
pd.set_option("display.max_colwidth", 2000)


class TreeStub:
    def __init__(self, x, y, cut_value):
        self.val = cut_value
        self.err = None
        self.left_y = None
        self.right_y = None
        self.left_err = None
        self.right_err = None
        self.left_c = None
        self.right_c = None

        self._get_err(x, y)

    def _get_err(self, x, y):
        left_mask = x <= self.val
        self.left_c = np.sum(left_mask * y) / np.sum(left_mask)
        self.left_y = y[: np.sum(left_mask)]
        self.left_err = np.sum(np.square(self.left_y - self.left_c))

        right_mask = x > self.val
        self.right_c = np.sum(right_mask * y) / np.sum(right_mask)
        self.right_y = y[np.sum(left_mask):]
        self.right_err = np.sum(np.square(self.right_y - self.right_c))

        self.err = self.left_err + self.right_err

    def get_residual(self, x, y):
        return y - self.predict(x)

    def predict(self, values):
        return (values < self.val) * self.left_c + (values >= self.val) * self.right_c


class BoostingTree:
    def __init__(self, m, x, y, candidates):
        self.m = m
        self.x = x
        self.y = y
        self.candidates = candidates
        self.errs = {}
        self.round_num = 3
        self.trees: List[TreeStub] = []

    def _get_min_loss_cut_candidate(self, x, y):

        min_ms = np.array([float("inf")])
        candidate_tree = None
        for value in self.candidates:
            tree_stub = TreeStub(x, y, value)
            self.errs.setdefault(value, []).append(np.round(tree_stub.err, self.round_num))
            if min_ms > tree_stub.err:
                min_ms = tree_stub.err
                candidate_tree = tree_stub

        self.errs.setdefault("loss", []).append(np.round(min_ms, 3))
        self.errs.setdefault("cut_value", []).append(candidate_tree.val)
        self.errs.setdefault("tree_left_value", []).append(np.round(candidate_tree.left_c, self.round_num))
        self.errs.setdefault("tree_right_value", []).append(np.round(candidate_tree.right_c, self.round_num))

        return candidate_tree

    def fit(self):
        y_iter = self.y
        for i in range(self.m):
            self.errs.setdefault("iter_num", []).append(i+1)
            tree_stub_i = self._get_min_loss_cut_candidate(self.x, y_iter)
            self.trees.append(tree_stub_i)
            y_iter = tree_stub_i.get_residual(self.x, y_iter)

        self._generate_summary()

    def predict(self, x):
        out = np.zeros_like(x, dtype=np.float32)
        for tree in self.trees:
            out += tree.predict(x)
        return out

    def _generate_summary(self):
        self.df = pd.DataFrame(self.errs)

    def summary(self):
        return self.df


def main():
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32)
    y = np.array([5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05], dtype=np.float32)
    # 切分点
    cut_candidate = np.array(range(3, 21, 2)) * 0.5
    # 定义boosting tree
    boostint_tree = BoostingTree(15, x, y, cut_candidate)
    # 训练
    boostint_tree.fit()
    # 展示训练结果
    print(boostint_tree.summary())
    #
    print("*" * 100)
    df = pd.DataFrame([y, boostint_tree.predict(x)])
    df['type'] = ['y', 'y_predict']
    print(df)


if __name__ == '__main__':
    main()
