"""
使用yield实现生成器，连接多个模块的原型机
"""
import os
import numpy as np
import time


def input_func():
    for i in range(10):
        yield i


class M1(object):
    @property
    def mi(self):
        return range(10)  # 10条数据

    def infer(self, input):
        for i in input:
            print('m1 infer')
            yield i + 5  # 经过m1，加5
            # yield f'm1_{i}'


class M2(object):
    def infer(self, input):
        for i in input:
            print('m2 infer')
            yield i * 2  # 经过m2，乘以2
            # yield f'm2_{i}'


def b_func():
    modules = [M1(), M2()]

    last_yield = None
    finnal_yield = None
    history = []
    for i, m in enumerate(modules):
        # print(f'model: {i}')
        input = last_yield if last_yield else m.mi
        ret = m.infer(input)
        if i == len(modules) - 1:
            finnal_yield = ret
        else:
            last_yield = ret
        history.append(input)
    history.append(finnal_yield)

    return finnal_yield, history


ret, his = b_func()
for i in range(len(his[0])):
    print(his[0][i], next(his[2]))
