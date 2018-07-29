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

# Qの初期化
V = np.zeros((N,N,N,N))

# 方策の読み込み
pi = np.load('pi.npy')


# 報酬を格納するリストの初期化
returns = [[[[[[] for s3 in range(N)] for s2 in range(N)]\
                    for s1 in range(N)] for s0 in range(N)] for action in range(num_action)]


reward_list = []
# 実行フェイズ
for epi in range(num_episode):
    print("episode#: %d" % epi)
    delta = 0

    done = False
    observation = env.reset() # 環境をリセット
    count =1
    tmp = 0
    while(done == False):
       s = [assign(min_list[i], max_list[i], N, observation[i]) for i in range(num_state)]
       action = np.argmax(pi[:,s[0],s[1],s[2],s[3]], axis=0)
       env.render()
       observation, reward, done, info = env.step(action)
       tmp += reward

    reward_list.append(tmp)

env.close()



