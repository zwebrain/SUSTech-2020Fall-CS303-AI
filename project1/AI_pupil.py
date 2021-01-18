import numpy as np
import random
import time

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
DIRECTION = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
INFINITY = 100000
WEIGHTS = [
    [120, -20, 20, 5, 5, 20, -20, 120],
    [-20, -40, -5, -5, -5, -5, -40, -20],
    [20, -5, 15, 3, 3, 15, -5, 20],
    [5, -5, 3, 3, 3, 3, -5, 5],
    [5, -5, 3, 3, 3, 3, -5, 5],
    [20, -5, 15, 3, 3, 15, -5, 20],
    [-20, -40, -5, -5, -5, -5, -40, -20],
    [120, -20, 20, 5, 5, 20, -20, 120]
]

MID_PHASE = 20
LAST_PHASE = 40


def isOnCorner(x, y):
    return (x == 0 or x == 7) and (y == 7 and y == 0)


def isOnEdge(x, y):
    return x == 0 or x == 7 or y == 7 or y == 0


def outOfBoard(x, y):
    return x < 0 or y < 0 or x > 7 or y > 7


def board_change(color, chessboard, this_move):
    new_chessboard = chessboard.copy()
    for d in DIRECTION:
        flag = False
        it = this_move[0] + d[0]
        jt = this_move[1] + d[1]
        if outOfBoard(it, jt):
            continue
        if new_chessboard[it][jt] == -color:
            flag = True
        while flag:
            it += d[0]
            jt += d[1]
            if outOfBoard(it, jt) or new_chessboard[it][jt] == COLOR_NONE:
                flag = False
                break
            elif new_chessboard[it][jt] == color:
                break
        if flag:
            it -= d[0]
            jt -= d[1]
            while it != this_move[0] or jt != this_move[1]:
                new_chessboard[it][jt] = color
                it -= d[0]
                jt -= d[1]
            new_chessboard[it][jt] = color
    return new_chessboard


def getScoreOfBoard(color, chessboard):
    idx = np.where(chessboard == color)
    idx = list(zip(idx[0], idx[1]))
    return len(idx)


def find_all_available(color, chessboard):
    available = np.zeros((8, 8))
    flag = False
    idx = np.where(chessboard == color)
    idx = list(zip(idx[0], idx[1]))
    for i, j in idx:
        for d in DIRECTION:
            score = 0
            it = i + d[0]
            jt = j + d[1]
            if outOfBoard(it, jt):
                continue
            if chessboard[it][jt] == -color:
                flag = True
                score = 1
            while flag:
                it += d[0]
                jt += d[1]
                if outOfBoard(it, jt) or chessboard[it][jt] == color:
                    flag = False
                    break
                elif chessboard[it][jt] == COLOR_NONE:
                    break
                else:
                    score += 1
            if flag:
                available[it][jt] += score
                flag = False
    idx = np.where(available != 0)
    res = list(zip(idx[0], idx[1]))
    return res


def mobility_count(color, chessboard):
    idx = np.where(chessboard == color)
    idx = list(zip(idx[0], idx[1]))
    if np.count_nonzero(chessboard) > 45 or len(idx) > 20:

        pass
    else:
        available = np.zeros((8, 8))
        flag = False
        for i, j in idx:
            for d in DIRECTION:
                score = 0
                it = i + d[0]
                jt = j + d[1]
                if outOfBoard(it, jt):
                    continue
                if chessboard[it][jt] == -color:
                    flag = True
                    score = 1
                while flag:
                    it += d[0]
                    jt += d[1]
                    if outOfBoard(it, jt) or chessboard[it][jt] == color:
                        flag = False
                        break
                    elif chessboard[it][jt] == COLOR_NONE:
                        break
                    else:
                        score += 1
                if flag:
                    available[it][jt] = 1
                    flag = False
        return np.count_nonzero(available)


def edge_stability(color, chessboard, corner, method):
    score = 0
    methods = [[1, 1, 1, 1], [1, 6, 1, -1], [6, 1, -1, 1], [6, 6, -1, -1]]
    m = methods[method]
    if chessboard[corner[0]][corner[1]] == color:
        score += 1
        i = m[0]
        while 7 > i > 0 and chessboard[i][corner[1]] == color:
            score += 1
            i += m[2]
        j = m[1]
        while 7 > j > 0 and chessboard[corner[0]][j] == color:
            score += 1
            j += m[3]
    return score


def stable_bonus(color, chessboard):
    score = 0
    corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
    for method, corner in enumerate(corners):
        score += edge_stability(color, chessboard, corner, method)
        score -= edge_stability(-color, chessboard, corner, method)
    return score


def frontier_bonus(color, chessboard):
    potential = 0
    for i in range(1, 7):
        for j in range(1, 7):
            if chessboard[i][j] is not COLOR_NONE:
                potential += (chessboard[i][j] * color) * (
                        abs(chessboard[i][j - 1]) + abs(chessboard[i][j + 1]) + abs(chessboard[i - 1][j]) +
                        abs(chessboard[i + 1][j]) + abs(chessboard[i - 1][j + 1]) + abs(chessboard[i - 1][j - 1]) +
                        abs(chessboard[i + 1][j - 1]) + abs(chessboard[i + 1][j + 1]) - 8)
    for j in range(1, 7):
        for i in [0, 7]:
            if chessboard[i][j] is not COLOR_NONE:
                potential += (chessboard[i][j] * color) * (abs(chessboard[i][j - 1]) + abs(chessboard[i][j + 1]) - 2)
    for i in range(1, 7):
        for j in [0, 7]:
            if chessboard[i][j] is not COLOR_NONE:
                potential += (chessboard[i][j] * color) * (abs(chessboard[i - 1][j]) + abs(chessboard[i + 1][j]) - 2)
    return potential


def corner_bonus(color, chessboard):
    score = 0
    for i in [0, 7]:
        for j in [0, 7]:
            if chessboard[i][j] is not COLOR_NONE:
                score += (chessboard[i][j] * color)
    return score


def disc_bonus(color, chessboard):
    score = 0
    for i in range(8):
        for j in range(8):
            if chessboard[i][j] == color:
                score += 1
            elif chessboard[i][j] == -color:
                score -= 1
    return score


def move_bonus(color, chessboard):
    mobility = len(find_all_available(color, chessboard)) - len(find_all_available(-color, chessboard))
    return mobility


def endgame_evl(color, chessboard):
    res = disc_bonus(color, chessboard)
    if res > 0:
        return INFINITY
    elif res < 0:
        return -INFINITY
    else:
        return evaluate(color, chessboard)


def place_bonus(color, chessboard):
    score = 0
    for i in range(8):
        for j in range(8):
            if chessboard[i][j] == color:
                score += WEIGHTS[i][j]
            elif chessboard[i][j] == -color:
                score -= WEIGHTS[i][j]
    return score


def evaluate(color, chessboard):
    score = 0
    score += (200 * stable_bonus(color, chessboard))
    score += (20 * frontier_bonus(color, chessboard))
    score += (100 * corner_bonus(color, chessboard))
    score += (10 * disc_bonus(color, chessboard))
    score += (50 * move_bonus(color, chessboard))
    score += (1 * place_bonus(color, chessboard))
    return score


def Max(color, chessboard, depth, alpha, beta):
    if depth == 0:
        return evaluate(color, chessboard)
    actions = find_all_available(color, chessboard)
    v = -INFINITY
    if len(actions) is 0:
        actions = find_all_available(-color, chessboard)
        if len(actions) is 0:
            return endgame_evl(color, chessboard)
        return Min(color, chessboard, depth, alpha, beta)
    for act in actions:
        new_chessboard = board_change(color, chessboard, act)
        v = max(v, Min(color, new_chessboard, depth - 1, alpha, beta))
        if v >= beta:
            alpha = max(alpha, v)
    return v


def Min(color, chessboard, depth, alpha, beta):
    if depth == 0:
        return evaluate(color, chessboard)
    actions = find_all_available(-color, chessboard)
    v = INFINITY
    if len(actions) is 0:
        actions = find_all_available(color, chessboard)
        if len(actions) is 0:
            return endgame_evl(color, chessboard)
        return Max(color, chessboard, depth, alpha, beta)
    for act in actions:
        new_chessboard = board_change(-color, chessboard, act)
        v = min(v, Max(color, new_chessboard, depth - 1, alpha, beta))
        if v <= alpha:
            return v
        beta = min(beta, v)
    return v


def alpha_beta(color, chessboard, depth):
    actions = find_all_available(color, chessboard)
    best_score = -INFINITY
    opt_action = None
    for act in actions:
        new_chessboard = board_change(color, chessboard, act)
        value = Min(color, new_chessboard, depth - 1, best_score, INFINITY)
        if value > best_score:
            best_score = value
            opt_action = act
    actions.append(opt_action)
    return actions


class Hashtable_Node:
    def __init__(self, lock, lower, upper, best_move, depth):
        self.lock = lock
        self.lower = lower
        self.upper = upper
        self.best_move = best_move
        self.depth = depth


class AI(object):
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        self.color = color
        self.time_out = time_out
        self.candidate_list = []

    def go(self, chessboard):
        self.candidate_list = alpha_beta(self.color, chessboard, 4)
