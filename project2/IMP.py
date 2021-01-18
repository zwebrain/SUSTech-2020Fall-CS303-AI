# -*- coding: utf-8 -*-
import math
import argparse
import random
import time
import multiprocessing as mp


class Graph:
    def __init__(self, node_len):
        self.node_len = node_len
        self.adj = [[] for _ in range(node_len + 1)]
        self.rev_adj = [[] for _ in range(node_len + 1)]

    def add_edge(self, u: int, v: int, t: float):
        self.adj[u].append((v, t))
        self.rev_adj[v].append((u, t))


def read_network_file(network_file):
    with open(network_file) as file:
        first_line = file.readline()
        node_len, edge_len = first_line.split(' ')
        network = Graph(int(node_len))
        for i in range(int(edge_len)):
            u, v, t = str.split(file.readline())
            network.add_edge(int(u), int(v), float(t))
        return network


def gamma(epsilon, delta):
    return (2 + 0.6666667 * epsilon) * math.log(1 / (delta * epsilon * epsilon))


def lgamma(n: int):
    result = 0
    for i in range(2, n + 1):
        result = result + math.log(i)
    return result


def lgcomb(n: int, k: int):
    return lgamma(n) - lgamma(k) - lgamma(n - k)


def judge(Sstar_k, rr):
    for s in Sstar_k:
        if s in rr:
            return True
    return False


def Max_Coverage(network: Graph, R, k):
    Sstar_k = set()
    theta = len(R)
    rr_count = [0] * (network.node_len + 1)
    find = dict()

    for i in range(theta):
        rr = R[i]
        for u in rr:
            rr_count[u] += 1
            if u in find:
                find[u].append(i)
            else:
                find[u] = [i]

    for i in range(k):
        u = rr_count.index(max(rr_count))
        if u == 0:
            z = i
            pos = 1
            for i in range(z, k):
                for x in range(pos, network.node_len + 1):
                    if x not in Sstar_k:
                        Sstar_k.add(x)
                        break
            break

        Sstar_k.add(u)
        rrs = find[u].copy()
        for rr in rrs:
            for v in R[rr]:
                find[v].remove(rr)
                rr_count[v] -= 1

    count = 0
    for rr in R:
        if judge(Sstar_k, rr):
            count += 1

    return Sstar_k, count * network.node_len / theta


def get_RR_IC(network: Graph, seed):
    RR_set = set(seed)
    active_set = set(seed)
    while active_set:
        new_active_set = set()
        for u in active_set:
            for v, t in network.rev_adj[u]:
                if v not in RR_set and v not in active_set:
                    prob = random.uniform(0.0, 1.0)
                    if prob <= t:
                        new_active_set.add(v)
        active_set = new_active_set
        RR_set = RR_set.union(new_active_set)
    return RR_set


def get_RR_LT(network: Graph, seed):
    RR_set = set(seed)
    active_set = set(seed)

    while active_set:
        new_active_set = set()
        for u in active_set:
            if len(network.rev_adj[u]) == 0:
                continue
            candidate = random.sample(network.rev_adj[u], 1)[0][0]
            if candidate not in RR_set:
                RR_set.add(candidate)
                new_active_set.add(candidate)
        active_set = new_active_set
    return RR_set


def get_RR(network: Graph, times: int, model):
    rr = []
    for _ in range(times):
        rr.append(get_RR_(network, model))
    return rr


def get_RR_(network: Graph, model):
    seed_list2 = {int(random.uniform(1, network.node_len + 1))}
    if model == 'IC':
        return get_RR_IC(network, seed_list2)
    else:
        return get_RR_LT(network, seed_list2)


def DSSA(network: Graph, epsilon, delta, k):
    Sstar_k = []
    n = network.node_len
    N_max = 7.5854467 / (3 + epsilon) * (2 + 0.6666667 * epsilon) * (lgcomb(n, k) + math.log(6 / delta)) * n / (
            epsilon * epsilon * k)
    t_max = int(math.log2(2 * N_max / gamma(epsilon, delta / 3)) + 1)
    t = 0
    lamb = gamma(epsilon, delta / 3 / t_max)
    totalSamples = int(lamb + 1)
    rtc = []
    rt = []

    rtf = []
    resm = 0
    for ttt in range(10):
        rtf.append([])
        for i in range(totalSamples):
            rtf[ttt].append(get_RR_(network, model))
            Sstar_k, itsk = Max_Coverage(network, rtf[ttt], k)
            if itsk > resm:
                resm = itsk
                rt = rtf[ttt]


    global core
    times = int(len(rt) / core)

    while totalSamples < N_max:
        t += 1
        if rtc is not None and len(rtc) > 0:
            rt = rt + rtc
        rtc = []

        pool = mp.Pool(core)
        result = []
        for _ in range(core):
            result.append(pool.apply_async(get_RR, args=(network, times, model)))
        pool.close()
        pool.join()
        for r in result:
            rtc.extend(r.get())

        for _ in range(len(rt) - len(rtc)):
            rtc.append(get_RR_(network, model))

        Sstar_k, itsk = Max_Coverage(network, rt, k)

        # eps1 = (itsk) / (itcsk) - 1
        # eps2 = epsilon * math.sqrt(n * (1 + epsilon) / (itcsk * math.pow(2, t - 1)))
        # eps3 = epsilon * math.sqrt(
        #     n * (1 + epsilon) * (0.63212056 - epsilon) /
        #     ((1 + epsilon / 3) * itcsk * math.pow(2, t - 1)))
        # if (eps1 + eps2 + eps1 * eps2) * (0.63212056 - epsilon) + \
        #         eps3 * 0.63212056 < epsilon:
        #     return Sstar_k
        global minmin
        if time.time() - start > time_limit / minmin:
            return Sstar_k

    return Sstar_k


def read_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--file_name', type=str, default='NetHEPT.txt')
    parser.add_argument('-k', '--seed', type=str, default='5')
    parser.add_argument('-m', '--model', type=str, default='LT')
    parser.add_argument('-t', '--time_limit', type=int, default=120)
    args = parser.parse_args()
    return args.file_name, args.seed, args.model, args.time_limit


if __name__ == '__main__':
    core = 8
    start = time.time()
    social_network_file, k, model, time_limit = read_arg()
    G = read_network_file(social_network_file)
    n = G.node_len
    minmin = 2.5

    epsilon = 0.001
    delta = 1 / n
    seeds = DSSA(G, epsilon, delta, int(k))

    for ss in seeds:
        print(ss)