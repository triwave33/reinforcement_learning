### -*-coding:utf-8-*-
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import grid_world


# agentの生成
agent = grid_world.Agent([0,0])

GAMMA = 0.9
ACTIONS = ['right', 'up', 'left', 'down']
num_row = 5 
num_col = 5
num_iteration = 100 # エピソード終了までの反復回数
num_episode = 10000 # MCサンプリング数

# Qの初期化
Q = np.zeros(len(ACTIONS), num_row, num_col))
# 方策の初期化
pi = np.random.randint(0,4, size=(5,5))
# 報酬を格納するリストの初期化
returns = [[[[] for col in range(num_col)] for row in range(num_row)] for a in range(len(ACTIONS))]


print("start iteration")
for epi in range(num_episode):
    print("episode#: %d" % epi)
    delta = 0
    # 開始点を設定
    i,j = np.random.randint(5, size=2)
    # 開始行動を設定
    first_action = np.random.randint(4)
    agent.set_pos([i,j]) # 移動前の状態に初期化
    tmp = 0
    # 初回のアクション 
    s = agent.get_pos()
    agent.move(ACTIONS[first_action]) # 移動
    reward = agent.reward(s, first_action)
    tmp += reward # 移動後の報酬を加算
    
    # 方策に従いエピソードを生成
    for k in range(1, num_iteration):
        s = agent.get_pos()
        action = pi[s[0],s[1]] 
        agent.move(ACTIONS[action]) # 移動
        reward = agent.reward(s, action)
        tmp += GAMMA**(k) * reward # 移動後の報酬を加算
    returns[first_action][i][j].append(tmp) # 収益をリストに追加

    
    Q =np.array([[[np.mean(returns[a][c][r]) for r in range(num_row)] \
                    for c in range(num_col)] for a in range(len(ACTIONS))])
    # piを更新
    pi = np.argmax(Q, axis=0)


print("length of returns")
returns_length =np.array([[[len(returns[a][c][r]) for r in range(num_row)] \
                for c in range(num_col)] for a in range(len(ACTIONS))])
print(returns_length)

print("Q")
print(Q)

