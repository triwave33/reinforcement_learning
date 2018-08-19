### -*-coding:utf-8-*-
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from tqdm import tqdm
import grid_world

   
# agentの生成
agent = grid_world.Agent([0,0])

ALPHA = 0.01
GAMMA = 0.9
ACTIONS = ['right', 'up', 'left', 'down']
num_row = 5 
num_col = 5
num_iteration = 100 # エピソード終了までの反復回数
num_episode = 10000 # サンプリング数
# piの設定
def pi(): # 状態sによらない方策
    return np.random.randint(0,4) # 上下左右がランダム


# Vの初期化
V = np.zeros((5,5))


print("start iteration")

count = 0

V_trend = np.zeros((num_episode, num_row, num_col))

for epi in tqdm(range(num_episode)):
    delta = 0
    i,j = np.random.randint(5, size=2)
    agent.set_pos([i,j]) # 移動前の状態に初期化
    s = agent.get_pos()
    for k in range(num_iteration):
        action = ACTIONS[np.random.randint(0,4)] 
        agent.move(action) # 移動
        s_dash = agent.get_pos() # 移動後の状態
        reward = agent.reward(s, action)
        V[s[0],s[1]] = V[s[0],s[1]] + ALPHA*(reward + GAMMA*V[s_dash[0], s_dash[1]]  - V[s[0],s[1]])
        s = agent.get_pos()


# 状態価値関数の表示
grid_world.V_value_plot(V) 
