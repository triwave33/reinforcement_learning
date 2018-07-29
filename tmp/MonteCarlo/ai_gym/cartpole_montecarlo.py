### -*-coding:utf-8-*-
import gym
from gym import spaces
import numpy as np
import time

env = gym.make('CartPole-v0')
env.reset()

# 状態空間を次の様に分割
N =5 # N分割
min_list = [-0.2, -2, -0.05, -1]
#max_list = [0.1, 1, 0.05, 1]
max_list = [0.15, 0.8, 0.025, 0.5]

# 状態変数を離散変数にアサインする関数
def assign(minimum, maximum, N, value):
    # 区間をN分割するためにはN+1点必要
    linspace = np.linspace(minimum, maximum, N+1)
    for i in range(N):
        if i ==0:
            low = -np.inf
        else:
            low = linspace[i]
        if i == N-1:
            high = np.inf
        else:
            high = linspace[i+1]
        if ((value >= low) & (value <= high)): # high側の等号はinf対策
            ret = i
    return ret


# 配列中に最大値を返すindexが複数ある場合、ランダムにindexを返す
def random_pick_argmax(array):
    max_indexes = np.where(array == array.max())[0]
    length = len(max_indexes)
    random_index = np.random.randint(length)
    return max_indexes[random_index]



GAMMA = 0.9
ACTIONS = [0,1]
EPSILON = 0.3
num_state = 4
num_action = 2
num_iteration = 10 # エピソード終了までの反復回数
num_episode = 10000 # MCサンプリング数
render = True # 描写モード

# Qの初期化
V = np.zeros((N,N,N,N))

# 方策の初期化
pi = np.random.rand(num_action,N,N,N,N)
pi = pi / np.sum(pi, axis=0) # 正規化


# 報酬を格納するリストの初期化
returns = [[[[[[] for s3 in range(N)] for s2 in range(N)]\
                    for s1 in range(N)] for s0 in range(N)] for action in range(num_action)]


return_list = []
# 学習フェイズ
for epi in range(num_episode):
    delta = 0

    done = False
    observation = env.reset() # 環境をリセット
    s = [assign(min_list[i], max_list[i], N, observation[i]) for i in range(num_state)]
    if render:
        env.render()

    s_list = [] # エピソードの状態履歴
    a_list = [] # エピソードの状態履歴
    r_list = [] # エピソードの状態履歴

    while(done == False):
        if render:
            env.render()

        s_list.append(s)
        action = np.random.choice([0,1], p = pi[:,s[0],s[1],s[2],s[3]])
        a_list.append(action)
        #print("action: %d" % action)
        observation, reward, done, info = env.step(action)
        r_list.append(reward)

        s = [assign(min_list[i], max_list[i], N, observation[i]) for i in range(num_state)]
    tmp = 0
    for i in range(len(s_list))[::-1]: # リストを逆側から検索
        tmp += r_list[i]
        returns[a_list[i]][s_list[i][0]][s_list[i][1]][s_list[i][2]][s_list[i][3]].append(tmp)
    return_list.append(tmp)
    print("eps: %.3f, epi#: %d, step_len: %d, reward: %3d" % (EPSILON, epi, len(r_list), tmp))

    
    returns_average =np.array([[[[[np.mean(returns[a][_s0][_s1][_s2][_s3]) \
                        for _s3 in range(N)] for _s2 in range(N)] \
                        for _s1 in range(N)] for _s0 in range(N)] for a in range(num_action)])
    # piを更新
    a_prime = np.argmax(returns_average, axis=0)
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



