# -*- coding: utf-8 -*-
import multiprocessing as mp
import time
import sys
import argparse
import random

import numpy as np


class Graph:
    def __init__(self, node_len):
        self.node_len = node_len
        self.adj = [[] for _ in range(node_len + 1)]
        self.rev_adj = [[] for _ in range(node_len + 1)]

    def add_edge(self, u: int, v: int, t: float):
        self.adj[u].append((v, t))
        self.rev_adj[v].append((u, t))

    def print(self):
        print(self.node_len)
        print(self.adj[10])
        print(self.rev_adj[10])


def ISE_IC(network: Graph, seed_list: list):
    active_set = seed_list.copy()
    activated = seed_list.copy()
    while active_set:
        u = active_set.pop(0)
        for v, t in network.adj[u]:
            if v not in activated:
                prob = random.random()
                if prob <= t:
                    activated.append(v)
                    active_set.append(v)
    return len(activated)


def ISE_LT(network: Graph, seed_list: list):
    active_set = seed_list.copy()
    activated = seed_list.copy()
    thresholds = np.random.rand(network.node_len + 1)
    weights = np.zeros(network.node_len + 1)
    while active_set:
        u = active_set.pop(0)
        for v, t in network.adj[u]:
            if v not in activated:
                weights[v] += t
                if weights[v] >= thresholds[v]:
                    activated.append(v)
                    active_set.append(v)
    return len(activated)


def read_network_file(network_file):
    with open(network_file) as file:
        first_line = file.readline()
        node_len, edge_len = first_line.split(' ')
        network = Graph(int(node_len))
        for i in range(int(edge_len)):
            u, v, t = str.split(file.readline())
            network.add_edge(int(u), int(v), float(t))
        return network


def read_seeds_file(seeds_file):
    seed_list = []
    with open(seeds_file) as file:
        lines = file.readlines()
        for line in lines:
            seed_list.append(int(line))
        return seed_list


def read_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--file_name', type=str, default='network.txt')
    parser.add_argument('-s', '--seed', type=str, default='network_seeds.txt')
    parser.add_argument('-m', '--model', type=str, default='IC')
    parser.add_argument('-t', '--time_limit', type=int, default=60)

    args = parser.parse_args()
    return args.file_name, args.seed, args.model, args.time_limit


def sampling(network: Graph, seed_list: list, selected, start, time_limit):
    res = 0.0
    cnt = 0
    limit = 0.8 * time_limit - 1
    for i in range(1000):
        if selected == 'IC':
            oneSample = ISE_IC(network, seed_list)
            res += oneSample
            cnt += 1
        else:
            oneSample = ISE_LT(network, seed_list)
            res += oneSample
            cnt += 1
        if time.time() - start > limit:
            break
    return res / cnt


if __name__ == '__main__':
    core = 8
    start = time.time()
    social_network_file, network_seeds_file, model, time_limit = read_arg()
    social_network = read_network_file(social_network_file)
    network_seeds = read_seeds_file(network_seeds_file)
    pool = mp.Pool(core)
    result = []
    for i in range(core):
        result.append(pool.apply_async(sampling, args=(social_network, network_seeds, model, start, time_limit)))
    pool.close()
    pool.join()
    res = 0
    for r in result:
        res += r.get()
    res /= len(result)
    print(res)
    sys.stdout.flush()
