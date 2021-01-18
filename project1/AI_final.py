from datetime import datetime
import numpy as np
import random

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
DIRECTION = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

WEIGHTS = [
    [130, -8, 2, 5, 5, 25, -100, 1300],
    [-8, -1500, -5, -5, -5, -5, -1500, -80],
    [25, -5, 15, 3, 3, 15, -5, 25],
    [10, -5, 3, 3, 3, 3, -5, 10],
    [10, -5, 3, 3, 3, 3, -5, 10],
    [25, -5, 15, 3, 3, 15, -5, 25],
    [-80, -1500, -5, -5, -5, -5, -1500, -100],
    [1300, -120, 27, 5, 5, 20, -100, 1300]
]
HASH_PLAYERS = np.zeros(2, dtype=np.int64)
HASH_PLAYERS[0] = random.randint(0, 9999999999999)
HASH_PLAYERS[1] = random.randint(0, 9999999999999)
HASH_CHESSBOARD_B = np.zeros((8, 8, 2), dtype=np.int64)
HASH_CHESSBOARD_W = np.zeros((8, 8, 2), dtype=np.int64)

for ih in range(8):
    for jh in range(8):
        HASH_CHESSBOARD_B[ih][jh][0] = random.randint(0, 9999999999999)
        HASH_CHESSBOARD_B[ih][jh][1] = random.randint(0, 9999999999999)
        HASH_CHESSBOARD_W[ih][jh][0] = random.randint(0, 9999999999999)
        HASH_CHESSBOARD_W[ih][jh][1] = random.randint(0, 9999999999999)


def getHashCode(color, chessboard):
    hashcode1 = 0
    hashcode2 = 0
    idx = np.where(chessboard == COLOR_BLACK)
    idx = list(zip(idx[0], idx[1]))
    for idx1 in idx:
        hashcode1 ^= HASH_CHESSBOARD_B[idx1[0]][idx1[1]][0]
        hashcode2 ^= HASH_CHESSBOARD_B[idx1[0]][idx1[1]][1]
    idx = np.where(chessboard == COLOR_WHITE)
    idx = list(zip(idx[0], idx[1]))
    for idx1 in idx:
        hashcode1 ^= HASH_CHESSBOARD_W[idx1[0]][idx1[1]][0]
        hashcode2 ^= HASH_CHESSBOARD_W[idx1[0]][idx1[1]][1]
    if color == COLOR_WHITE:
        hashcode1 ^= HASH_PLAYERS[0]
        hashcode2 ^= HASH_PLAYERS[1]
    t = (hashcode1, hashcode2)
    return t


class Hash_Table_Node:
    def __init__(self, alpha, beta, best_move, depth):
        self.alpha = alpha
        self.beta = beta
        self.best_move = best_move
        self.depth = depth


def isOnCorner(x, y):
    return (x == 0 or x == 7) and (y == 7 and y == 0)


def isOnEdge(x, y):
    return x == 0 or x == 7 or y == 7 or y == 0


def isONC(x, y):
    return (x == 1 or x == 6) and (y == 1 and y == 6)


def outOfBoard(x, y):
    return x < 0 or y < 0 or x > 7 or y > 7


def getScoreOfBoard(color, chessboard):
    idx = np.where(chessboard == color)
    idx = list(zip(idx[0], idx[1]))
    return len(idx)


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


class AI(object):
    three = [1, 3, 9, 27, 81, 243, 729, 2187]
    INFINITY = 10000000
    find = {
        -6560: 10, -2186: 7, -6558: 7, -2184: 5, -726: 4, -2178: 4,
        3886: -8, 4354: -8, -4351: -8, -3157: -8, -3565: -5, -2263: -5,
        -4294: -6, -2266: -6, -2193: 7, -1459: 7, -3055: -3, -3109: -3,
        -3648: -5, - 736: -5, 3648: -5, 736: -5, 4351: 7, 3157: 7,
        -2209: -5, -3403: -5, -2185: -9, -4371: -9, 2185: -8, 4371: -8,
        -2194: -6, -3646: -6, 2194: 4, -3646: 4, 2188: -4, -4210: -7, -4318: -7,
        -3481: -4, -4315: -4, 3481: -4, 4315: -4, 4210: -7, 4318: -7, -4372: -2,
        0: -10, 2187: -9, 1: -8, 2: -9, 1458: -9, 6: -9, 486: -8, 18: -8,
        -2350: 6, -2242: 6, 2350: 8, 2242: 8, 2206: 7, 2604: 7, -2206: 3, -2604: 3,
        6319: 6, 2213: 6,
    }

    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        self.color = color
        self.time_out = time_out
        self.candidate_list = []
        self.counter = 0
        self.time = datetime.now()
        self.hashtable = {}

    def hash_update(self, hashcode, alpha, beta, opt_action, depth):
        self.hashtable[(hashcode[0], hashcode[1])] = Hash_Table_Node(alpha, beta, opt_action, depth)

    def move_bonus(self, chessboard, opponent=-1):
        score1 = 0
        score2 = 0
        list1 = self.find_all_available(chessboard)
        list2 = self.find_all_available(chessboard, opponent=1)
        l1 = len(list1)
        l2 = len(list2)
        if l2 < 5:
            score1 += 2
            for i in list2:
                if not isONC(i[0], i[1]):
                    break
                score1 += 10
        if opponent == 1:
            score1 *= 2
        if l1 < 5:
            score2 -= 2
            for i in list1:
                if not isONC(i[0], i[1]):
                    break
                score2 -= 15
        if opponent == -1:
            score2 *= 2
        return score1 + score2 + l1 - l2

    def edge_stability(self, chessboard, corner, method):
        score = 0
        methods = [[1, 1, 1, 1], [1, 6, 1, -1], [6, 1, -1, 1], [6, 6, -1, -1]]
        m = methods[method]
        if chessboard[corner[0]][corner[1]] == self.color:
            score += 2
            i1 = m[0]
            while 7 > i1 > 0 and chessboard[i1][corner[1]] == self.color:
                score += 1
                i1 += m[2]
            j1 = m[1]
            while 7 > j1 > 0 and chessboard[corner[0]][j1] == self.color:
                score += 1
                j1 += m[3]

            min_step = min((i1 - m[0]) / m[2], (j1 - m[1]) / m[3])
            if min_step > 0:
                if chessboard[m[0]][m[1]] == self.color:
                    score += 1.5
                    if min_step > 1:
                        if chessboard[m[0]][m[1] + m[3]] == self.color and chessboard[m[0] + m[2]][m[1]] == self.color:
                            score += 2.5
                            if min_step > 2:
                                if chessboard[m[0]][m[1] + 2 * m[3]] == self.color and chessboard[m[0] + 2 * m[2]][
                                    m[1]] == self.color and chessboard[m[0] + m[2]][m[1] + m[3]] == self.color:
                                    score += 7

        elif chessboard[corner[0]][corner[1]] == -self.color:
            score -= 2
            i1 = m[0]
            while 7 > i1 > 0 and chessboard[i1][corner[1]] == -self.color:
                score -= 1
                i1 += m[2]
            j1 = m[1]
            while 7 > j1 > 0 and chessboard[corner[0]][j1] == -self.color:
                score -= 1
                j1 += m[3]
            min_step = min((i1 - m[0]) / m[2], (j1 - m[1]) / m[3])
            if min_step > 0:
                if chessboard[m[0]][m[1]] == -self.color:
                    score -= 1.5
                    if min_step > 1:
                        if chessboard[m[0]][m[1] + m[3]] == -self.color and chessboard[m[0] + m[2]][
                            m[1]] == -self.color:
                            score -= 2.5
                            if min_step > 2:
                                if chessboard[m[0]][m[1] + 2 * m[3]] == -self.color and chessboard[m[0] + 2 * m[2]][
                                    m[1]] == -self.color and chessboard[m[0] + m[2]][m[1] + m[3]] == -self.color:
                                    score -= 7
        return score

    def move_level(self, chessboard, act, opponent=-1):
        value = 0
        if (act[0] in [0, 7]) ^ (act[1] in [0, 7]):
            if act[0] == 0:
                for k in range(8):
                    value += (self.color * chessboard[0][k] + 1) * self.three[k]
            elif act[0] == 7:
                for k in range(8):
                    value += (self.color * chessboard[7][k] + 1) * self.three[k]
            elif act[1] == 0:
                for k in range(8):
                    value += (self.color * chessboard[k][0] + 1) * self.three[k]
            elif act[1] == 0:
                for k in range(8):
                    value += (self.color * chessboard[k][7] + 1) * self.three[k]
            value = value * opponent
            if value in self.find:
                return self.find[value]
        return 0

    def stable_bonus(self, chessboard):
        score = 0
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        for method, corner in enumerate(corners):
            score += self.edge_stability(chessboard, corner, method)
        return score

    def frontier_bonus(self, chessboard):
        potential = 0
        for i in range(1, 7):
            for j in range(1, 7):
                if chessboard[i][j] is not COLOR_NONE:
                    potential += (chessboard[i][j] * self.color) * (
                            abs(chessboard[i][j - 1]) + abs(chessboard[i][j + 1]) + abs(chessboard[i - 1][j]) +
                            abs(chessboard[i + 1][j]) + abs(chessboard[i - 1][j + 1]) + abs(chessboard[i - 1][j - 1]) +
                            abs(chessboard[i + 1][j - 1]) + abs(chessboard[i + 1][j + 1]) - 8)
        for j in range(1, 7):
            for i in [0, 7]:
                if chessboard[i][j] is not COLOR_NONE:
                    potential += (chessboard[i][j] * self.color) * (
                            abs(chessboard[i][j - 1]) + abs(chessboard[i][j + 1]) - 2)
        for i in range(1, 7):
            for j in [0, 7]:
                if chessboard[i][j] is not COLOR_NONE:
                    potential += (chessboard[i][j] * self.color) * (
                            abs(chessboard[i - 1][j]) + abs(chessboard[i + 1][j]) - 2)
        return potential

    def corner_bonus(self, chessboard):
        score = 0
        for i in [0, 7]:
            for j in [0, 7]:
                if chessboard[i][j] is not COLOR_NONE:
                    score += chessboard[i][j]
        if self.color == -1:
            score = -score
        return score

    def disc_bonus(self, chessboard):
        score = 0
        for i in range(8):
            for j in range(8):
                if chessboard[i][j] == self.color:
                    score += 1
                elif chessboard[i][j] == -self.color:
                    score -= 1
        return score

    @staticmethod
    def logistics(num):
        return num

    def endgame_evl(self, chessboard):
        res = self.disc_bonus(chessboard)
        if res > 0:
            return self.INFINITY
        else:
            return -self.INFINITY

    def place_bonus(self, chessboard):
        score = 0
        for i in range(8):
            for j in range(8):
                if chessboard[i][j] == self.color:
                    score += WEIGHTS[i][j]
                elif chessboard[i][j] == -self.color:
                    score -= WEIGHTS[i][j]
        return score

    def evaluate(self, chessboard, act):
        w = [1150, 215, 40, 4, 30, 4]
        if self.counter > 23:
            w = [1200, 175, 5, 23, 10, 0]
        score = 0
        score += self.logistics(w[0] * self.stable_bonus(chessboard))
        score += self.logistics(w[1] * self.move_level(chessboard, act))
        score += self.logistics(w[2] * self.frontier_bonus(chessboard))
        score += self.logistics(w[3] * self.disc_bonus(chessboard))
        score += self.logistics(w[4] * self.move_bonus(chessboard))
        if w[5] != 0:
            score += self.logistics(w[5] * self.place_bonus(chessboard))
        return score

    def evaluate_o(self, chessboard, act):
        w = [1200, 175, 5, 23, 10, 0]
        if self.counter > 23:
            w = [1200, 175, 5, 23, 10, 0]
        score = 0
        score += self.logistics(w[0] * self.stable_bonus(chessboard))
        score += self.logistics(w[1] * self.move_level(chessboard, act, 1))
        score += self.logistics(w[2] * self.frontier_bonus(chessboard))
        score += self.logistics(w[3] * self.disc_bonus(chessboard))
        score += self.logistics(w[4] * self.move_bonus(chessboard))
        if w[5] != 0:
            score += self.logistics(w[5] * self.place_bonus(chessboard))
        return score

    @staticmethod
    def find_all_method_A(chessboard, color, idx):
        available = np.zeros((8, 8))
        flag = False
        for i1, j1 in idx:
            for d in DIRECTION:
                score = 0
                it = i1 + d[0]
                jt = j1 + d[1]
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

    @staticmethod
    def find_all_method_B(chessboard, color):
        available = np.zeros((8, 8))
        idx = np.where(chessboard == COLOR_NONE)
        idx = list(zip(idx[0], idx[1]))
        flag = False
        for i1, j1 in idx:
            for d in DIRECTION:
                it = i1 + d[0]
                jt = j1 + d[1]
                if outOfBoard(it, jt):
                    continue
                if chessboard[it][jt] == -color:
                    flag = True
                while flag:
                    it += d[0]
                    jt += d[1]
                    if outOfBoard(it, jt) or chessboard[it][jt] == COLOR_NONE:
                        flag = False
                        break
                    elif chessboard[it][jt] == color:
                        available[i1][j1] += 1
                        break
                if flag:
                    flag = False
                    break
        idx = np.where(available != 0)
        res = list(zip(idx[0], idx[1]))
        return res

    def find_all_available(self, chessboard, opponent=0):
        color = self.color
        if opponent is not 0:
            color = -self.color
        idx = np.where(chessboard == color)
        idx = list(zip(idx[0], idx[1]))
        if len(idx) < 19 or self.counter < 19:
            return self.find_all_method_A(chessboard, color, idx)
        else:
            return self.find_all_method_B(chessboard, color)

    def alpha_beta_with_hash(self, chessboard, alpha, beta, depth):
        hashcode = getHashCode(self.color, chessboard)
        if hashcode in self.hashtable:
            node = self.hashtable[hashcode]
            d = node.depth
            if d <= depth:
                if node.alpha > alpha:
                    alpha = node.alpha
                    if alpha >= beta:
                        return alpha
                if node.beta < beta:
                    beta = node.beta
                    if beta <= alpha:
                        return beta
        best_value, opt_action = self.alpha_beta(chessboard, alpha, beta, depth)
        if best_value >= beta:
            self.hash_update(hashcode, best_value, self.INFINITY, opt_action, depth)
        elif best_value <= alpha:
            self.hash_update(hashcode, -self.INFINITY, best_value, opt_action, depth)
        else:
            self.hash_update(hashcode, best_value, best_value, opt_action, depth)

    def alpha_beta(self, chessboard, alpha, beta, depth):
        actions = self.candidate_list
        best_score = alpha
        opt_action = self.candidate_list[-1]
        new_chessboards = []
        evaluate_list = []

        for i, act in enumerate(actions):
            new_chessboard = board_change(self.color, chessboard, act)
            new_chessboards.append(new_chessboard)
            val = self.evaluate(new_chessboard, act)
            if val >= self.INFINITY:
                return val, act
            evaluate_list.append((val, i))

        evaluate_list.sort(reverse=True)
        for i in range(min(len(new_chessboards), 8)):
            evaluate = evaluate_list[i]
            new_chessboard = new_chessboards[evaluate[1]]
            value = self.Min(new_chessboard, depth - 1, best_score, beta, evaluate[0])
            if value >= best_score:
                best_score = value
                opt_action = actions[evaluate[1]]
                self.candidate_list.append(opt_action)
        return best_score, opt_action

    def Max(self, chessboard, depth, alpha, beta, cur_val, actions=None):
        scale = 8
        tim = (datetime.now() - self.time).seconds
        if depth == 0:
            return cur_val
        if depth == 1:
            if tim < 1:
                scale = 11
            elif tim > 4:
                return cur_val
            elif tim > 3.7:
                scale = 3
            elif tim > 3.1:
                scale = 5

        hashcode = getHashCode(self.color, chessboard)
        if hashcode in self.hashtable:
            node = self.hashtable[hashcode]
            d = node.depth
            if d <= depth:
                if node.alpha > alpha:
                    alpha = node.alpha
                    if alpha >= beta:
                        return alpha
                if node.beta < beta:
                    beta = node.beta
                    if beta <= alpha:
                        return beta

        if actions is None:
            actions = self.find_all_available(chessboard)
        best_value = -self.INFINITY
        if len(actions) is 0:
            actions = self.find_all_available(chessboard, opponent=1)
            if len(actions) is 0:
                return self.endgame_evl(chessboard)
            return self.Min(chessboard, depth, alpha, beta, cur_val, actions)

        new_chessboards = []
        evaluate_list = []
        for i, act in enumerate(actions):
            new_chessboard = board_change(self.color, chessboard, act)
            new_chessboards.append(new_chessboard)
            val = self.evaluate(new_chessboard, act)
            if val >= self.INFINITY:
                return val
            evaluate_list.append((val, i))

        opt_action = None
        evaluate_list.sort(reverse=True)
        for i in range(min(len(new_chessboards), scale)):
            new_chessboard = new_chessboards[evaluate_list[i][1]]
            value = self.Min(new_chessboard, depth - 1, best_value, beta, evaluate_list[i][0])
            if value > best_value:
                best_value = value
                opt_action = actions[evaluate_list[i][1]]
            if best_value >= beta:
                return best_value
            alpha = max(alpha, best_value)
            if best_value >= beta:
                self.hash_update(hashcode, best_value, self.INFINITY, opt_action, depth)
            elif best_value <= alpha:
                self.hash_update(hashcode, -self.INFINITY, best_value, opt_action, depth)
            else:
                self.hash_update(hashcode, best_value, best_value, opt_action, depth)
        return best_value

    def Min(self, chessboard, depth, alpha, beta, cur_val, actions=None):
        scale = 8
        tim = (datetime.now() - self.time).seconds
        if depth == 0:
            return cur_val
        if depth == 1:
            if tim < 1:
                scale = 11
            elif tim > 4:
                return cur_val
            elif tim > 3.7:
                scale = 3
            elif tim > 3.1:
                scale = 5
        hashcode = getHashCode(-self.color, chessboard)
        if hashcode in self.hashtable:
            node = self.hashtable[hashcode]
            d = node.depth
            if d <= depth:
                if node.alpha > alpha:
                    alpha = node.alpha
                    if alpha >= beta:
                        return alpha
                if node.beta < beta:
                    beta = node.beta
                    if beta <= alpha:
                        return beta

        if actions is None:
            actions = self.find_all_available(chessboard, opponent=1)
        best_value = self.INFINITY
        if len(actions) is 0:
            actions = self.find_all_available(chessboard)
            if len(actions) is 0:
                return self.endgame_evl(chessboard)
            return self.Max(chessboard, depth, alpha, beta, cur_val, actions)

        new_chessboards = []
        evaluate_list = []
        for i, act in enumerate(actions):
            new_chessboard = board_change(-self.color, chessboard, act)
            new_chessboards.append(new_chessboard)
            val = self.evaluate_o(new_chessboard, act)
            if val <= -self.INFINITY:
                return val
            evaluate_list.append((val, i))
        opt_action = None
        evaluate_list.sort()
        for i in range(min(len(new_chessboards), scale)):
            new_chessboard = new_chessboards[evaluate_list[i][1]]
            value = self.Max(new_chessboard, depth - 1, best_value, beta, evaluate_list[i][0])
            if value < best_value:
                best_value = value
                opt_action = actions[evaluate_list[i][1]]
            if best_value <= alpha:
                return best_value
            beta = min(beta, best_value)
            if best_value >= beta:
                self.hash_update(hashcode, best_value, self.INFINITY, opt_action, depth)
            elif best_value <= alpha:
                self.hash_update(hashcode, -self.INFINITY, best_value, opt_action, depth)
            else:
                self.hash_update(hashcode, best_value, best_value, opt_action, depth)

        return best_value

    def go(self, chessboard):
        self.candidate_list.clear()
        self.counter += 1
        actions = self.find_all_available(chessboard)
        self.candidate_list.extend(actions)
        if len(actions) > 0 and actions[-1] is not None:
            if self.counter > 23:
                dep = (self.counter / 2 - 7)
                self.alpha_beta_with_hash(chessboard=chessboard, alpha=-self.INFINITY, beta=self.INFINITY, depth=dep)
            elif len(actions) > 17:
                self.alpha_beta_with_hash(chessboard=chessboard, alpha=-self.INFINITY, beta=self.INFINITY, depth=3)
            elif len(actions) > 8:
                self.alpha_beta_with_hash(chessboard=chessboard, alpha=-self.INFINITY, beta=self.INFINITY, depth=4)
            else:
                self.alpha_beta_with_hash(chessboard=chessboard, alpha=-self.INFINITY, beta=self.INFINITY, depth=5)
