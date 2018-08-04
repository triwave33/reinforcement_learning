### -*-coding:utf-8-*-
import gym
from gym import spaces
import numpy as np
import time
import matplotlib.pyplot as plt

# Ai GymのCartPoleを使用
env = gym.make('CartPole-v0')
env.reset()

# 状態空間を次の様に分割
N =5 # N分割
min_list = [-0.2, -2, -0.05, -1] # 下限値
max_list = [0.15, 0.8, 0.025, 0.5] # 上限値


ACTIONS = [0,1]
EPSILON_INITIAL = 0.3
num_state = 4
num_action = 2
num_episode = 2000 # MCサンプリング数
render = False # 描写モード
state = 'learning' # learning or playing

# Vの初期化
V = np.zeros((N,N,N,N))

# 方策の初期化
pi = np.random.rand(num_action,N,N,N,N)
pi = pi / np.sum(pi, axis=0) # 正規化


# 報酬を格納するリストの初期化
returns = [[[[[[] for s3 in range(N)] for s2 in range(N)]\
                    for s1 in range(N)] for s0 in range(N)] for action in range(num_action)]
return_list = [] # エピソードごとの収益を格納（状態は気にしない）


for epi in range(num_episode*2): 
    if epi == num_episode:
        state = 'playing'
    EPSILON = max(0, EPSILON_INITIAL * (1.0 -epi*1.0/num_episode))
    delta = 0

    done = False
    observation = env.reset() # 環境をリセット
    # 状態を離散化
    s = [np.digitize(observation[i], np.linspace(min_list[i], max_list[i], N-1))\
        for i in range(num_state)]
    if render:
        env.render()

    s_list = [] # エピソードの状態履歴
    a_list = [] # エピソードの状態履歴
    r_list = [] # エピソードの状態履歴
    
    # エピソードを終端までプレイ
    while(done == False):
        if render:
            env.render()

        s_list.append(s)
        action = np.random.choice([0,1], p = pi[:,s[0],s[1],s[2],s[3]])
        a_list.append(action)
        #print("action: %d" % action)
        observation, reward, done, info = env.step(action)
        r_list.append(reward)

        # 状態を離散化
        s = [np.digitize(observation[i], np.linspace(min_list[i], \
            max_list[i], N-1)) for i in range(num_state)]

    # エピソードの各ステップから終端状態までの報酬をリストに格納
    tmp = 0
    for i in range(len(s_list))[::-1]: # リストを逆側から検索
        tmp += r_list[i]
        returns[a_list[i]][s_list[i][0]][s_list[i][1]][s_list[i][2]][s_list[i][3]].append(tmp)
    return_list.append(tmp)
    print("eps: %.3f, epi#: %d, state: %s, reward: %3d" % (EPSILON, epi, state, tmp))

    
    Q = np.array([[[[[np.mean(returns[a][_s0][_s1][_s2][_s3]) \
                   for _s3 in range(N)] for _s2 in range(N)] \
                   for _s1 in range(N)] for _s0 in range(N)] for a in range(num_action)])
    # epsilon-greedyに従いpiを更新
    a_prime = np.argmax(Q, axis=0)
    for i0 in range(N):
        for i1 in range(N):
            for i2 in range(N):
                for i3 in range(N):
                    for a in range(num_action):
                        if a == a_prime[i0,i1,i2,i3]:
                            pi[a,i0,i1,i2,i3] = 1 - EPSILON + EPSILON/num_action
                        else:
                            pi[a,i0,i1,i2,i3] = EPSILON/num_action

    #print(pi[:,:,:,2,2])

env.close()

returns_length =np.array([[[[[len(returns[a][_s0][_s1][_s2][_s3]) \
                    for _s3 in range(N)] for _s2 in range(N)] \
                    for _s1 in range(N)] for _s0 in range(N)] for a in range(num_action)])


# 学習の経過をプロット
plt.plot(return_list)
plt.show()
