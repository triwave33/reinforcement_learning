### -*-coding:utf-8-*-
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from tqdm import tqdm
import  gym
import math

# Ai GymのCartPoleを使用
env = gym.make('CartPole-v0')

# 状態空間を次の様に分割
N =8 # N分割
min_list = [-5, -2, -0.5, -1] # 下限値
max_list = [5, 2, 0.5, 1] # 上限値



ACTIONS = [0,1]
ALPHA = 0.2
ALPHA_INI = 0.2
BETA = 0.2
BETA_INI = 0.2
GAMMA = 1.
EPS_INI = 1.
num_state = 4
num_action = 2
num_episode = 3000 # サンプリング数
render = 0 # 描写モード
ex_factor = 1.1 # epsilonがゼロになったあとも学習を続けるパラメータ

return_list = []

# Qの初期化
Q = np.zeros(((num_action,) + (N,)*4))

# Qの初期化
rho = 0

# epsilon-greedyのための確率分布を初期化
p = np.random.rand(num_action)
p = p / np.sum(p, axis=0) # 正規化


def select_action(Q, s, eps):
    # e-greedyによる行動選択   
    greedy = np.argmax(Q[:,s[0],s[1],s[2],s[3]])
    p = [(1- eps + eps/len(ACTIONS)) if i == greedy \
        else eps/len(ACTIONS) for i in range(len(ACTIONS))]
    return  np.random.choice(np.arange(len(ACTIONS)), p=p)
   

#for epi in tqdm(range(int(num_episode*ex_factor))):
for epi in range(int(num_episode*ex_factor)):
    # greedy方策を徐々に確定的にしていく
    EPSILON = max(0, EPS_INI* (1- epi*1./num_episode))
    ALPHA = max(0, ALPHA_INI* (1- epi*1./num_episode))
    BETA = max(0, BETA_INI* (1- epi*1./num_episode))
    
    # sの初期化
    done = False
    observation = env.reset() # 環境をリセット
    # 状態を離散化
    s = [int(np.digitize(observation[i], np.linspace(min_list[i],\
            max_list[i], N-1))) for i in range(num_state)]
    
    tmp = 0 # 報酬積算用
    count = 0
    # エピソードを終端までプレイ
    while(done==False):
        if render:
            env.render()
        
        # epsilon greedyで行動選択
        a = select_action(Q, s, EPSILON)

        # 行動aをとり、r, s'を観測
        observation, reward, done, info = env.step(a)

        # 状態を離散化
        s_dash = [int(np.digitize(observation[i], np.linspace(min_list[i], \
            max_list[i], N-1))) for i in range(num_state)]
        
        tmp += reward

        # argmaxによる行動選択(Q_learning)   
        Q_dash = np.max(Q[:,s_dash[0],s_dash[1],s_dash[2],s_dash[3]])

        # Qの更新
        Q[tuple([a]) + tuple(s)] += ALPHA*(reward -rho + (Q_dash)\
                                    - Q[tuple([a]) + tuple(s)])
        
        maxQ = np.max(Q[:,s[0],s[1],s[2],s[3]])
        if Q[tuple([a]) + tuple(s)] == maxQ:
            #print("Q(s,a) == maxQ(s,a)")
            rho += BETA*(reward-rho+Q_dash- maxQ)

        count += 1

        s = s_dash
    
    return_list.append(tmp)
    print("epi: %d, eps#: %.3f, reward: %3d" % (epi, EPSILON, tmp))
    env.close()


while(False):
    done = False
    observation = env.reset() # 環境をリセット
    # 状態を離散化
    s = [int(np.digitize(observation[i], np.linspace(min_list[i],\
            max_list[i], N-1))) for i in range(num_state)]
    # e-greedyによる行動選択   
    a = select_action(Q, s, EPSILON)
    while(done == False):
        # 行動aをとり、r, s'を観測
        observation, reward, done, info = env.step(a)
        env.render()

        # 状態を離散化
        s_dash = [int(np.digitize(observation[i], np.linspace(min_list[i], \
            max_list[i], N-1))) for i in range(num_state)]

        # e-greedyによる行動選択   
        a = select_action(Q, s_dash, 0)


