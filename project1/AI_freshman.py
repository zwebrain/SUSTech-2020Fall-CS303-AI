import numpy as np
import random
import time

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
Direction = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
random.seed(0)


class AI(object):
    # chessboard_size, color, time_out passed from agent
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        self.color = color
        self.time_out = time_out
        self.candidate_list = []

    def find_all_available(self, chessboard):
        available = np.zeros((8, 8))
        flag = False
        idx = np.where(chessboard == self.color)
        idx = list(zip(idx[0], idx[1]))
        for i, j in idx:
            for d in Direction:
                score = 0
                it = i + d[0]
                jt = j + d[1]
                if it < 0 or jt < 0 or it > 7 or jt > 7:
                    continue
                if chessboard[it][jt] == -self.color:
                    flag = True
                    score = 1
                while flag:
                    it += d[0]
                    jt += d[1]
                    if it < 0 or jt < 0 or it > 7 or jt > 7 or chessboard[it][jt] == self.color:
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
        self.candidate_list.extend(res)
        self.candidate_list.append(res[int(random.random() * (len(res)))])
        '''
        if len(res) > 3:
            # self.candidate_list.append(res[np.random.randint(0, len(res))])
            if len(np.where(available != COLOR_NONE)) > 10:
                min_score = np.min(available[np.nonzero(available)])
                idx_op = np.where(available == min_score)
            else:
                max_score = np.max(available[np.nonzero(available)])
                idx_op = np.where(available == max_score)
            self.candidate_list.extend(list(zip(idx_op[0], idx_op[1])))
        if random.random() > 0.9 and len(res) > 3:
            self.candidate_list.append(res[np.random.randint(0, len(res))])
        '''

    def go(self, chessboard):
        self.candidate_list.clear()
        self.find_all_available(chessboard)

    # ==============Find new pos========================================
    # Make sure that the position of your decision in chess board is empty.
    # If not, the system will return error.
    # Add your decision into candidate_list, Records the chess board
    # You need add all the positions which is valid
    # candidate_list example: [(3,3),(4,4)]
    # You need append your decision at the end of the candidate_list,
    # we will choose the last element of the candidate_list as the position you choose
    # If there is no valid position, you must return a empty list.
