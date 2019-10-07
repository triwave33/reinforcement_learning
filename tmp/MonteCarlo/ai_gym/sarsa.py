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
GAMMA = 1.
EPS_INI = 1.
num_state = 4
num_action = 2
num_episode = 2000 # サンプリング数
render = 0 # 描写モード
ex_factor = 1.1 # epsilonがゼロになったあとも学習を続けるパラメータ

# Qの初期化
Q = np.zeros(((num_action,) +(N,)*4))

# epsilon-greedyのための確率分布
p = np.random.rand(num_action)
p = p / np.sum(p, axis=0) # 正規化

sarsa_10times = np.zeros((10, int(num_episode*ex_factor)))

def select_action(Q, s, eps):
    # e-greedyによる行動選択   
    greedy = np.argmax(Q[:,s[0],s[1],s[2],s[3]])
    p = [(1- eps + eps/len(ACTIONS)) if i == greedy \
        else eps/len(ACTIONS) for i in range(len(ACTIONS))]
    return  np.random.choice(np.arange(len(ACTIONS)), p=p)
   

for trial in range(10):
    #for epi in tqdm(range(int(num_episode*ex_factor))):
    for epi in range(int(num_episode*ex_factor)):
        # greedy方策を徐々に確定的にしていく
        EPSILON = max(0, EPS_INI* (1- epi*1./num_episode))
        
        # sの初期化
        done = False
        observation = env.reset() # 環境をリセット
        # 状態を離散化
        s = [int(np.digitize(observation[i], np.linspace(min_list[i],\
                max_list[i], N-1))) for i in range(num_state)]
        # e-greedyによる行動選択   
        a = select_action(Q, s, EPSILON)
        
        tmp = 0 # 報酬積算用
        count = 0
        # エピソードを終端までプレイ
        while(done==False):
            if render:
                env.render()
    
            # 行動aをとり、r, s'を観測
            observation, reward, done, info = env.step(a)
    
            tmp += reward
    
            # 状態を離散化
            s_dash = [int(np.digitize(observation[i], np.linspace(min_list[i], \
                max_list[i], N-1))) for i in range(num_state)]
    
            # e-greedyによる行動選択   
            a_dash = select_action(Q, s_dash, EPSILON)
    
            # Qの更新
            Q_dash = Q[tuple([a_dash]) + tuple(s_dash,)]
            Q[tuple([a]) + tuple(s)] += ALPHA*(reward + GAMMA*(Q_dash)\
                                        - Q[tuple([a]) + tuple(s)])
            s = s_dash
            a = a_dash
    
            count += 1
    
        print("epi: %d, eps#: %.3f, reward: %3d" % (epi, EPSILON, tmp))
        sarsa_10times[trial, epi] = tmp
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
        a = select_action(Q, s_dash, EPSILON)


