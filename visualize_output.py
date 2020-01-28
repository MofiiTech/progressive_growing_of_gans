#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Filename : visualize_output.py
# @Date : 2019-12-02
# @Author : Wufei Ma

import os
import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

output_file = 'train_out_dec_01.txt'


def get_time(time_list):
    if len(time_list) == 1:
        return int(time_list[0][:-1])
    elif len(time_list) == 2:
        tmp = int(time_list[1][:-1])
        return tmp + 60 * int(time_list[0][:-1])
    elif len(time_list) == 3:
        tmp = int(time_list[2][:-1])
        tmp += 60 * int(time_list[1][:-1])
        return tmp + 3600 * int(time_list[0][:-1])
    elif len(time_list) == 4:
        tmp = int(time_list[3][:-1])
        tmp += 60 * int(time_list[2][:-1])
        tmp += 3600 * int(time_list[1][:-1])
        return tmp + 24 * 3600 * int(time_list[0][:-1])


if __name__ == '__main__':

    cur_time = []
    sec_per_kimg = []

    f = open(output_file, 'r')
    for line in f:
        line = line.strip()
        if not line.startswith('tick'):
            continue
        tokens = line.split()

        idx1 = tokens.index('sec/kimg')
        idx2 = tokens.index('time')
        idx3 = tokens.index('sec/tick')
        sec_per_kimg.append(tokens[idx1+1])
        cur_time.append(get_time(tokens[idx2+1: idx3]))

    sec_per_kimg = np.array(sec_per_kimg, dtype=np.float32)
    cur_time = np.array(cur_time, dtype=np.int32)
    x = np.arange(1, len(sec_per_kimg) + 1)

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 3, 1)

    ax.plot(x, sec_per_kimg)
    ax.set_xlabel('tick')
    ax.set_ylabel('sec/kimg')

    kimg_per_hr = 3600 / sec_per_kimg
    ax = fig.add_subplot(1, 3, 2)
    ax.plot(x, kimg_per_hr)
    ax.set_xlabel('tick')
    ax.set_ylabel('kimg/hr')

    remaining_time = cur_time + sec_per_kimg * (15000 - x)
    ax = fig.add_subplot(1, 3, 3)
    ax.plot(x, remaining_time)
    ax.set_xlabel('tick')
    ax.set_ylabel('remaining time (sec)')

    plt.savefig('train_out_dec_01.png', dpi=200)

    print(kimg_per_hr[-10:])
