import numpy as np
import random
from . import AI_final as AI1
from . import AI_final as AI2

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(0)
Direction = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
chessboard = np.zeros((8, 8))
chessboard[3][4] = 1
chessboard[4][3] = 1
chessboard[3][3] = -1
chessboard[4][4] = -1

'''
idx0 = np.where(chessboard != 0)
idx0 = list(zip(idx0[0], idx0[1]))
print(idx0)
for i0, j0 in idx0:
    print(i0, j0)
'''


def board_change(this_chessboard, this_place, color):
    for d in Direction:
        flag = False
        (it, jt) = (this_place[0] + d[0], this_place[1] + d[1])
        if it < 0 or jt < 0 or it > 7 or jt > 7:
            continue
        if this_chessboard[it][jt] == -color:
            flag = True
        while flag:
            it += d[0]
            jt += d[1]
            if it < 0 or jt < 0 or it > 7 or jt > 7 or this_chessboard[it][jt] == COLOR_NONE:
                flag = False
                break
            elif this_chessboard[it][jt] == color:
                break
        if flag:
            it -= d[0]
            jt -= d[1]
            while it != this_place[0] or jt != this_place[1]:
                this_chessboard[it][jt] = color
                it -= d[0]
                jt -= d[1]
            this_chessboard[it][jt] = color
    return this_chessboard


def print_chessboard(now_chessboard, clist):
    token = now_chessboard.copy()
    token = token.astype(str)
    token[token == '-1.0'] = '●'
    token[token == '1.0'] = '○'
    token[token == '0.0'] = '十'
    if len(clist) > 0 and clist[-1] is not None:
        for cl in clist:
            token[cl[0]][cl[1]] = '※'
    a = np.array(['零', '一', '二', '三', '四', '五', '六', '七'])
    b = np.array(['❤', '零', '一', '二', '三', '四', '五', '六', '七'])
    token = np.insert(token, 0, values=a, axis=0)
    token = np.insert(token, 0, values=b, axis=1)
    print(token)


AIb = AI1(8, COLOR_BLACK, 5)
AIw = AI2(8, COLOR_WHITE, 5)
players = [AIb, AIw]
player_flag = 0
game_flag = True

while game_flag:
    # print('')
    # print('')

    players[player_flag].go(chessboard)

    candidate = players[player_flag].candidate_list
    if len(candidate) == 0 or candidate[-1] is None:
        print('You pass')
        players[1 - player_flag].go(chessboard)
        candidate2 = players[1 - player_flag].candidate_list
        if len(candidate2) == 0 or candidate2[-1] is None:
            print('Game Over')
            print_chessboard(chessboard, candidate)
            idx = np.where(chessboard == COLOR_WHITE)
            res = list(zip(idx[0], idx[1]))
            print('white numbers are: {}'.format(len(res)))
            idx = np.where(chessboard == COLOR_BLACK)
            res = list(zip(idx[0], idx[1]))
            print('black numbers are: {}'.format(len(res)))
            game_flag = False
    else:
        print_chessboard(chessboard, candidate)

        if player_flag == 0:
            print('You are ● player')
        else:
            print('You are ○ player')
        print('Your optional list is as follows: {}'.format(candidate))
        print('')
        print('')
        place = players[player_flag].candidate_list[-1]

        while place not in candidate:
            print('There is something wrong with your input!')
            place = eval(input())
        print('You have chosen: {}'.format(place))

        chessboard = board_change(chessboard, place, players[player_flag].color)
    player_flag = 1 - player_flag
