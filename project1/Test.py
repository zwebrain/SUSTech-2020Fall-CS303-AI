import numpy as np

import random

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(0)
Direction = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
chessboard = np.zeros((8, 8))
chessboard[3][4] = -1
chessboard[4][3] = -1
chessboard[3][3] = 1
chessboard[4][4] = 1

HASH_PLAYERS = np.zeros(2, dtype=np.int64)
HASH_PLAYERS[0] = random.randint(100000000000, 999999999999)
HASH_PLAYERS[1] = random.randint(100000000000, 999999999999)
HASH_ZOBRIST_FLIP = np.zeros((8, 8, 2), dtype=np.int64)
HASH_CHESSBOARD_B = np.zeros((8, 8, 2), dtype=np.int64)
HASH_CHESSBOARD_W = np.zeros((8, 8, 2), dtype=np.int64)

for i in range(8):
    for j in range(8):
        HASH_CHESSBOARD_B[i][j][0] = random.randint(100000000000, 999999999999) ^ HASH_PLAYERS[0]
        HASH_CHESSBOARD_B[i][j][1] = random.randint(100000000000, 999999999999) ^ HASH_PLAYERS[1]
        HASH_CHESSBOARD_W[i][j][0] = random.randint(100000000000, 999999999999) ^ HASH_PLAYERS[0]
        HASH_CHESSBOARD_W[i][j][1] = random.randint(100000000000, 999999999999) ^ HASH_PLAYERS[1]
        HASH_ZOBRIST_FLIP[i][j][0] = HASH_CHESSBOARD_B[i][j][0] ^ HASH_CHESSBOARD_W[i][j][0]
        HASH_ZOBRIST_FLIP[i][j][1] = HASH_CHESSBOARD_B[i][j][1] ^ HASH_CHESSBOARD_W[i][j][1]
MID_PHASE = 20
LAST_PHASE = 40


def getHashCode(color, board):
    hashcode = [0, 0]
    idx = np.where(board == COLOR_BLACK)
    idx = list(zip(idx[0], idx[1]))
    for idx1 in idx:
        hashcode[0] = hashcode[0] ^ HASH_CHESSBOARD_B[idx1[0]][idx1[1]][0]
        hashcode[1] ^= HASH_CHESSBOARD_B[idx1[0]][idx1[1]][1]
    idx = np.where(board == COLOR_WHITE)
    idx = list(zip(idx[0], idx[1]))
    for idx1 in idx:
        hashcode[0] ^= HASH_CHESSBOARD_W[idx1[0]][idx1[1]][0]
        hashcode[1] ^= HASH_CHESSBOARD_W[idx1[0]][idx1[1]][1]
    if color == COLOR_WHITE:
        hashcode[0] ^= HASH_PLAYERS[0]
        hashcode[1] ^= HASH_PLAYERS[1]
    return hashcode


def changeHashCode(color, is_pass, action, flips, hashcode):
    if is_pass:
        hashcode[0] ^= HASH_PLAYERS[0]
        hashcode[1] ^= HASH_PLAYERS[1]
        return hashcode
    if color == COLOR_BLACK:
        hashcode[0] ^= HASH_CHESSBOARD_B[action[0]][action[1]][0]
        hashcode[1] ^= HASH_CHESSBOARD_B[action[0]][action[1]][1]
    else:
        hashcode[0] ^= HASH_CHESSBOARD_W[action[0]][action[1]][0]
        hashcode[1] ^= HASH_CHESSBOARD_W[action[0]][action[1]][1]
    for f in flips:
        hashcode[0] ^= HASH_ZOBRIST_FLIP[f[0]][f[1]][0]
        hashcode[1] ^= HASH_ZOBRIST_FLIP[f[0]][f[1]][1]
    return hashcode


print(HASH_CHESSBOARD_B)
print(HASH_CHESSBOARD_W)
print(chessboard)
h1 = getHashCode(COLOR_WHITE, chessboard)
h2 = getHashCode(COLOR_BLACK, chessboard)

chessboard[2][3] = -1
chessboard[3][3] = -1
print(chessboard)
h4 = getHashCode(COLOR_BLACK, chessboard)
h5 = getHashCode(COLOR_WHITE, chessboard)
print(h1)
print(h2)
print(h4)
print(h5)
