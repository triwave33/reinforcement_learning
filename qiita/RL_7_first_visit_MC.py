### -*-coding:utf-8-*-
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from tqdm import tqdm
import grid_world

   
# agentの生成
agent = grid_world.Agent([0,0])

GAMMA = 0.9
ACTIONS = ['right', 'up', 'left', 'down']
num_row = 5 
num_col = 5
num_iteration = 100 # エピソード終了までの反復回数
num_episode = 10000 # MCサンプリング数
# piの設定
def pi(): # 状態sによらない方策
    return np.random.randint(0,4) # 上下左右がランダム


# Vの初期化
V = np.zeros((5,5))

# 報酬を格納するリストの初期化
returns = [[[] for c in range(num_col)] for r in range(num_row)]

print("start iteration")

count = 0

N = 100
V_trend = np.zeros((N, num_row, num_col))

for epi in tqdm(range(num_episode)):
    delta = 0
    i,j = np.random.randint(5, size=2)
    agent.set_pos([i,j]) # 移動前の状態に初期化
    tmp = 0
    
    # 1エピソードの履歴を保存するリスト
    s_list = []
    a_list = []
    r_list = []
    
    # エピソードを探索
    for k in range(num_iteration):
        action = ACTIONS[np.random.randint(0,4)] 
        s = agent.get_pos()
        agent.move(action) # 移動
        s_dash = agent.get_pos() # 移動後の状態
        reward = agent.reward(s, action)
        tmp += GAMMA**(k) * reward # 移動後の報酬を加算
        # リストに格納
        s_list.append(s)
        a_list.append(action)
        r_list.append(reward) # 全体の収益

    returns[i][j].append(tmp) 
    #tmp2 = 0
    #for i in range(len(s_list))[::-1]: # リストを逆側から検索
    #    tmp2 += r_list[i]
    #    returns[s_list[i][0]][s_list[i][1]].append(tmp) 

    

returns_length =np.array([[len(returns[r][c]) for c in range(num_col)] \
                    for r in range(num_row)])
print("length of returns")
print returns_length

returns_average =np.array([[np.mean(returns[r][c]) for c in range(num_col)] \
                    for r in range(num_row)])
print("average of returns")
print(returns_average)

V = returns_average

## 状態価値関数を表示。状態関数は行動価値関数の行動に関しての期待をとる
#ax = plt.gca()
#plt.xlim(0,5)
#plt.ylim(0,5)
#ax.xaxis.set_ticklabels([])
#ax.yaxis.set_ticklabels([])
#V = returns_average
#V_round = np.round(V, decimals=2)
#for i in range(5):
#    for j in range(5):
#        # rect
#        rect = plt.Rectangle(xy =(i,j) , width=1, height=1, fill=False)
#        ax.add_patch(rect)
#        # 座標のインデックスの調整
#        x = -j-1 
#        y = i
#        # text
#        plt.text(i+ 0.4, j+0.5, "%s" % (str(V_round[x,y])))
#plt.savefig('output_files/RL_7_v_pi.png')
#plt.close()
#

