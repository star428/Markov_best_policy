#!/usr/bin/env python
# encoding: utf-8
"""
@author: star428
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited 
@contact: yewang863@gmail.com
@software: pycharm
@file: Markov_use_bellman.py
@time: 2020/11/30 20:42
@desc:
"""
import numpy as np

symbol = ['#', 1, -1]


class map:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.my_map = {}
        self.make_map()

    def make_map(self):
        prim_value = 0  # 最初每个点的最初效用值
        prim_action = None  # 记录最后的最优action
        for index_x in range(1, self.x + 1):
            for index_y in range(1, self.y + 1):
                self.my_map[(index_x, index_y)] = [prim_value,
                                                   prim_action]  # 初始化地图

    def put_wall(self, x, y):
        self.my_map[(x, y)] = ['#', '#']

    def put_destination(self, x, y, value):
        self.my_map[(x, y)] = [value, value]

    def change_value(self, new_value):
        for index in new_value.keys():
            self.my_map[index][0] = new_value[index]

    def print_map(self):

        for index_y in range(map.y, 0, -1):
            temp = []
            for index_x in range(1, map.x + 1):
                if self.my_map[(index_x, index_y)][1] == '#':
                    temp.append("#")
                elif self.my_map[(index_x, index_y)][1] == 1:
                    temp.append(1)
                elif self.my_map[(index_x, index_y)][1] == -1:
                    temp.append(-1)
                else:
                    temp.append(self.my_map[(index_x, index_y)][1])

            print(temp)

    def print_value(self):
        for index_y in range(map.y, 0, -1):
            temp = []
            for index_x in range(1, map.x + 1):
                if self.my_map[(index_x, index_y)][1] == '#':
                    temp.append("#")
                elif self.my_map[(index_x, index_y)][1] == 1:
                    temp.append(1)
                elif self.my_map[(index_x, index_y)][1] == -1:
                    temp.append(-1)
                else:
                    temp.append(self.my_map[(index_x, index_y)][0])

            print(temp)


def up(x, y, map):
    if y + 1 <= map.y and not (map.my_map[(x, y + 1)][0] is symbol[0]):
        y += 1
    return x, y


def down(x, y, map):
    if y - 1 > 0 and not (map.my_map[(x, y - 1)][0] is symbol[0]):
        y -= 1
    return x, y


def left(x, y, map):
    if x - 1 > 0 and not (map.my_map[(x - 1, y)][0] is symbol[0]):
        x -= 1
    return x, y


def right(x, y, map):
    if x + 1 <= map.x and not (map.my_map[(x + 1, y)][0] is symbol[0]):
        x += 1
    return x, y


def make_reward(map, value):
    reward = {}
    for index_x in range(1, map.x + 1):
        for index_y in range(1, map.y + 1):
            if not (map.my_map[(index_x, index_y)][0] in symbol):
                reward[(index_x, index_y)] = value

    return reward


def bellman_function(x, y, map, reward, a1, a2, a3, b):
    array = [a1 * map.my_map[up(x, y, map)][0] + \
             a2 * map.my_map[left(x, y, map)][0] + \
             a3 * map.my_map[right(x, y, map)][0],
             # ======================
             a1 * map.my_map[down(x, y, map)][0] + \
             a2 * map.my_map[left(x, y, map)][0] + \
             a3 * map.my_map[right(x, y, map)][0],
             # ======================
             a1 * map.my_map[left(x, y, map)][0] + \
             a2 * map.my_map[up(x, y, map)][0] + \
             a3 * map.my_map[down(x, y, map)][0],
             # ======================
             a1 * map.my_map[right(x, y, map)][0] + \
             a2 * map.my_map[up(x, y, map)][0] + \
             a3 * map.my_map[down(x, y, map)][0]
             # ======================
             ]
    new_value = reward[(x, y)] + b * np.max(array)
    return new_value


def iteration_of_value(map, reward, a1, a2, a3, b):
    new_value = {}
    stop = False
    i = 0
    while not stop:
        for index_x in range(1, map.x + 1):
            for index_y in range(1, map.y + 1):
                if not (map.my_map[(index_x, index_y)][1] in symbol):
                    new_value[(index_x, index_y)] = \
                        round_point_five(bellman_function(index_x, index_y, map,
                                                          reward,
                                                          a1, a2, a3, b))

        if the_equal(new_value, map):
            stop = True
        i += 1
        # print(i)
        if i == 10000:
            stop = True
        map.change_value(new_value)

    return map


def round_point_five(num):
    return int(num * 100000) / 100000


def the_equal(new_value, map):
    result = True
    for key in new_value.keys():
        if map.my_map[key][0] != new_value[key]:
            result = False
            break
    return result


def choose_best_way(map, a1, a2, a3):
    for x in range(1, map.x + 1):
        for y in range(1, map.y + 1):
            if not (map.my_map[(x, y)][1] in symbol):
                array = [a1 * map.my_map[up(x, y, map)][0] + \
                         a2 * map.my_map[left(x, y, map)][0] + \
                         a3 * map.my_map[right(x, y, map)][0],
                         # ======================
                         a1 * map.my_map[down(x, y, map)][0] + \
                         a2 * map.my_map[left(x, y, map)][0] + \
                         a3 * map.my_map[right(x, y, map)][0],
                         # ======================
                         a1 * map.my_map[left(x, y, map)][0] + \
                         a2 * map.my_map[up(x, y, map)][0] + \
                         a3 * map.my_map[down(x, y, map)][0],
                         # ======================
                         a1 * map.my_map[right(x, y, map)][0] + \
                         a2 * map.my_map[up(x, y, map)][0] + \
                         a3 * map.my_map[down(x, y, map)][0]
                         # ======================
                         ]
                max = np.max(array)
                best_way = []
                for index in range(4):
                    if array[index] == max:
                        best_way.append(num_to_direction(index))
                map.my_map[(x, y)][1] = best_way


def num_to_direction(num):
    direction = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    return direction[num]


if __name__ == "__main__":
    map = map(5, 5)  # 生成x * y地图
    # 放置障碍
    map.put_wall(2, 2)
    map.put_wall(4, 3)
    # 放置目标值
    map.put_destination(5, 5, 1)
    map.put_destination(5, 4, -1)
    # 计算回报函数（输入的为回报函数的值）
    reward = make_reward(map, -1)
    # 计算最终的效用值(输入agent的各个方向的概率和折扣因子）
    map = iteration_of_value(map, reward, 0.8, 0.1, 0.1, 0.5)
    # 计算最优路径
    choose_best_way(map, 0.8, 0.1, 0.1)
    # 打印相关数据
    map.print_map()
    map.print_value()
