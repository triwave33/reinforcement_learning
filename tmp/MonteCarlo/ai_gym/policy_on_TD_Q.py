### -*-coding:utf-8-*-
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from tqdm import tqdm
import  gym

# Ai GymのCartPoleを使用
env = gym.make('CartPole-v0')

# 状態空間を次の様に分割
N =5 # N分割
min_list = [-0.2, -2, -0.05, -1] # 下限値
max_list = [0.15, 0.8, 0.025, 0.5] # 上限値


ACTIONS = [0,1]
ALPHA = 0.2
GAMMA = 1.0
EPS_INI = 1.0
num_state = 4
num_action = 2
num_episode = 5000 # サンプリング数
render = 0 # 描写モード
state = 'learning' # learning or playing
ex_factor = 1.2 # epsilonがゼロになったあとも学習を続けるパラメータ

# Qの初期化
Q = np.zeros((num_action,N,N,N,N))
Q_trend = np.zeros(int(num_episode*ex_factor))

# epsilon-greedyのための確率分布
p = np.random.rand(num_action)
p = p / np.sum(p, axis=0) # 正規化

count = 0
return_list = []

for epi in tqdm(range(int(num_episode*ex_factor))):
    # greedy方策を徐々に確定的にしていく
    EPSILON = max(0, EPS_INI* (1- epi*1./num_episode))
    
    # sの初期化
    done = False
    observation = env.reset() # 環境をリセット
    # 状態を離散化
    s = [np.digitize(observation[i], np.linspace(min_list[i],\
            max_list[i], N-1)) for i in range(num_state)]
    # e-greedyによる行動選択   
    greedy = np.argmax(Q[:,s[0],s[1],s[2],s[3]])
    p = [(1- EPSILON + EPSILON/len(ACTIONS)) if i == greedy \
        else EPSILON/len(ACTIONS) for i in range(len(ACTIONS))]
    a = np.random.choice(np.arange(len(ACTIONS)), p=p)
    
    tmp = 0 # 報酬積算用
    # エピソードを終端までプレイ
    while(done==False):
        if render:
            env.render()

        # 行動aをとり、r, s'を観測
        observation, reward, done, info = env.step(a)
        tmp += reward
        # 状態を離散化
        s_dash = [np.digitize(observation[i], np.linspace(min_list[i], \
            max_list[i], N-1)) for i in range(num_state)]
        greedy = np.argmax(Q[:,s_dash[0],s_dash[1],s_dash[2],s_dash[3]])
        p = [(1- EPSILON + EPSILON/len(ACTIONS)) if i == greedy \
            else EPSILON/len(ACTIONS) for i in range(len(ACTIONS))]
        # e-greedyによる行動選択   
        a_dash = np.random.choice(np.arange(len(ACTIONS)), p=p)
        # Qの更新
        Q[a,s[0],s[1],s[2],s[3]] = Q[a,s[0],s[1],s[2],s[3]] \
                                    + ALPHA*(reward+GAMMA*np.max(Q[:,\
                                     s_dash[0],s_dash[1],s_dash[2],s_dash[3]]) \
                                    - Q[a,s[0],s[1],s[2],s[3]])
        s = s_dash
        a = a_dash
    return_list.append(tmp)
    Q_trend[epi] = Q[0,2,3,2,2]
   
env.close()

plt.plot(return_list)
plt.title("returns")
plt.show()

plt.plot(Q_trend)
plt.title("Q_trend for a certain state")
plt.show()

print("Q")
print Q

V = np.max(Q, axis=0)
print("V")
print V


## Vのヒストグラムをプロット
#for i in range(num_row):
#    for j in range(num_col):
#        plt.subplot(5,5, i*5+j+1)
#        plt.hist(returns[i][j])
#
#plt.show()

