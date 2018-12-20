### -*-coding:utf-8-*-
import random
import numpy as np
import scipy.linalg as LA
from itertools import product
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tqdm import tqdm
import  gym
import math
import seaborn as sns
import os
import datetime
import shutil
import sys
from collections import deque
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.optimizers import Adam, SGD, RMSprop
from keras import backend as K
import tensorflow as tf
import time


# 学習用パラメータ
ACTIONS = [0,1,2]
GAMMA = 0.99
EPS_LAST = 0
render = 0 # 描画モード
num_episode = 1601

# ファイル保存用パラメータ
fileOut = True
now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
path = './' # 保存先フォルダ(フォルダが自動生成される)
#path = '/volumes/data/dataset/ai_gym/'
os.mkdir(path +now)
os.mkdir(path +now + '/fig')
os.mkdir(path +now + '/theta')
myfile = os.path.dirname(os.path.abspath(__file__)) + '/true_online_sarsa_wx_mountaincar.py'
shutil.copyfile(myfile, path +now + '/theta.setting.py')
draw_all_action = False

# 画像ファイル描画用パラメータ
meshgrid = 25 # 描画のグリッド間隔（学習とは関係無い）
rec_interval = 500000 # 画像を保存するステップ間隔
one_step_record = False

# Mountaincar
game = 'MountainCar-v0'
env = gym.make('MountainCar-v0')
NUM_STATE = 2
NUM_ACTION = 3


from gym import wrappers
#env = wrappers.Monitor(env, "/Users/tu_MB12/Documents/github/reinforcement_learning/qiita", video_callable=(lambda ep: ep % 1600 == 0))


class TableAgent:
    def __init__(self,N,alpha_ini,alpha_last,eps_ini,eps_last):    
        self.N = N # テーブルの分割数
        self.done = False 
        self.Q = np.zeros(((NUM_ACTION,) + (self.N,)*NUM_STATE))  
        self.reward_array = [] 
        self.min_list = np.array([-1.2, -0.07])  # グリッド分割下限値
        self.max_list = np.array([0.6,0.07])  # グリッド分割上限値
        self.alpha_ini = alpha_ini 
        self.alpha_last = alpha_last
        self.eps_ini = eps_ini
        self.eps_last = eps_last
 
    def digitize(self,obs): # envからの観測値をを離散化
        s = [int(np.digitize(obs[i], np.linspace(self.min_list[i], \
                self.max_list[i], self.N-1))) for i in range(NUM_STATE)]
        return s

    def getQ(self,s,a): # s, aにおける行動価値関数を出力
        return self.Q[a,s[0],s[1]]

    def select_action(self, s, eps):
        s = self.digitize(s)
        # e-greedyによる行動選択   
        if np.random.rand < eps: # random
            action = np.random.randint(NUM_ACTION)
            return action
        else:
            action = np.argmax(self.Q[:,s[0],s[1]]) # greedy
            # 最大値をとるアクションが複数ある場合、その中からランダムに選択
            is_greedy_index = np.where(self.Q[:,s[0],s[1]] == action)[0]
            if len(is_greedy_index) > 1:
                action = np.random.choice(is_greedy_index)
            return action

    def train(self,s,a,reward,s_dash,a_dash):
        s = self.digitize(s)
        s_dash = self.digitize(s_dash)
        Qval = self.getQ(s,a)
        Qval_dash = self.getQ(s_dash,a_dash)
        self.Q[a,s[0],s[1]] = Qval + ALPHA * (reward + GAMMA * Qval_dash - Qval)

    def calc_Z(self): # 画像保存の値を出力
        x = np.linspace(self.min_list[0],self.max_list[0],meshgrid)
        y = np.linspace(self.min_list[1],self.max_list[1],meshgrid)
        X,Y = np.meshgrid(x,y)
        Z = np.array([[[agent.getQ(self.digitize(np.array([i,j])),k) for i in x] for j in y] for k in range(NUM_ACTION)])
        return X, Y, Z


class LinearFuncAgent:
    # 基底関数
    min_list = env.observation_space.low  # 下限値 [-1.2 -0.07]
    max_list = env.observation_space.high  # 上限値 [-0.6, 0.07]
    norm_factor = np.array([1.2, 0.07]) # 状態間のスケールを調整するをファクター
    norm_factor = norm_factor.reshape(len(norm_factor),1)


    def __init__(self,N, sigma, alpha_ini, alpha_last, eps_ini, eps_last):
        self.done = False
        self.s1_space = np.linspace(self.min_list[0], self.max_list[0], N)
        self.s2_space = np.linspace(self.min_list[1], self.max_list[1], N)
        b = (N**NUM_STATE) # 状態空間を分割した場合の総数
        self.alpha_ini = alpha_ini
        self.alpha_last = alpha_last
        self.eps_ini = eps_ini
        self.eps_last = eps_last

        # 基底関数の定数項を初期化（学習対象外）
        # 基底関数をガウス関数とし、中心値µを初期化
        self.mu_array = np.random.rand(NUM_STATE, b) # ランダムの場合
        #self.mu_array = np.zeros((NUM_STATE, b)) # オール0の場合
        cnt =0
        for i in self.s1_space:
            for j in self.s2_space:
                self.mu_array[0,cnt] =i
                self.mu_array[1,cnt] =j
                cnt+=1
        # 分散を初期化（固定かつ共通）
        self.sigma=0.1

        # 3つのアクションに対して、同じ基底関数セットを用いる
        self.mu_list = [np.copy(self.mu_array)] * NUM_ACTION
        
        # 学習対象のパラメータの初期化
        self.theta_list = [np.zeros(b), np.zeros(b), np.zeros(b)]


    # 基底関数(入力:状態空間(2次元)、出力:基底関数の出力(b*3次元))
    def rbfs(self, s):
        s = s.reshape(len(s),1) # 2次元に整形
        return np.exp(-np.square(LA.norm((self.mu_list-s)/self.norm_factor, axis=1))/(2*self.sigma**2)) 

        
    def getQ(self, s, a):
        # Q = X.T.dot(Theta)
        return (self.rbfs(s)[a]).dot(self.theta_list[a])


    def select_action(self, s, eps):
        # e-greedyによる行動選択   
        if np.random.rand() < eps: # random
            action = np.random.randint(NUM_ACTION)
            return action
        else:
            qs = [self.getQ(s,i) for i in range(NUM_ACTION)]
            action = np.argmax(qs)
            # 最大値をとる行動が複数ある場合はさらにランダムに選択
            is_greedy_index = np.where(qs == action)[0]
            if len(is_greedy_index) > 1:
                action = np.random.choice(is_greedy_index)
            return action

    def train(self, s, a, reward, s_dash, a_dash):
        X = self.rbfs(s)
        Q_val = self.getQ(s,a)
        Q_val_dash = self.getQ(s_dash,a_dash)

        DELTA = reward + GAMMA*Q_val_dash - Q_val
        # パラメータの更新
        self.theta_list[a] = self.theta_list[a] + ALPHA * DELTA * X[a]

    def calc_Z(self):
        x = np.linspace(self.min_list[0],self.max_list[0],meshgrid)
        y = np.linspace(self.min_list[1],self.max_list[1],meshgrid)
        X,Y = np.meshgrid(x,y)
        Z = np.array([[[agent.getQ(np.array([i,j]),k) for i in x] for j in y] for k in range(NUM_ACTION)])
        return X, Y, Z


   
class DqnAgent:
    num_batch = 16
    num_memory = 10000
    min_list = env.observation_space.low  # 下限値 [-1.2 -0.07]
    max_list = env.observation_space.high  # 上限値 [-0.6, 0.07]

    def __init__(self, lr, h1, h2, TAU, num_memory, in_dim, out_dim, eps_ini, eps_last, use_clip_reward):
        self.done = False
        self.lr = lr    # learning rate
        self.h1 = h1    # 1st hidden layer
        self.h2 = h2    # 2nd hidden layer
        self.TAU = TAU
        self.in_dim = (in_dim,) # NNモデル生成のために入力変数を指定
        self.out_dim = out_dim  # NNモデル生成のために出力変数を指定 
        self.eps_ini = eps_ini
        self.eps_last = eps_last
        self.alpha_ini = 0 # dummy # 他のクラスとの共通化のためダミー
        self.alpha_last = 0 # dummy
        self.use_clip_reward = use_clip_reward
        self.buff = ReplayBuffer(self.num_memory)
        self.model = self.build_model() # モデルを生成
        self.targetModel = self.build_model()
        self.targetModel.set_weights(self.model.get_weights())
        
        optimizer = RMSprop(self.lr)
        
        self.model.compile(loss=self.huberloss, optimizer=optimizer, metrics = ['accuracy'])

    def huberloss(self,y_true, y_pred):
        err = y_true - y_pred
        cond = K.abs(err) < 1.0
        L2 = 0.5 * K.square(err)
        L1 = (K.abs(err) - 0.5)
        loss = tf.where(cond, L2, L1)  # Keras does not cover where function in tensorflow :-(
        return K.mean(loss)

    def build_model(self):
        model = Sequential()
        model.add(Dense(self.h1, input_shape = self.in_dim, activation='relu'))
        model.add(Dense(self.h2, activation='relu'))
        model.add(Dense(self.out_dim, activation='linear', init='zero'))
        model.summary()
        return model

    def getQ(self, s, a):
        s = s.reshape(1, len(s)) # 2次元配列に整形
        q = self.model.predict(s)
        return self.model.predict(s)[0][a]

    def select_action(self, s, eps):
        # e-greedyによる行動選択   
        if np.random.rand() < eps: # random
            action = np.random.randint(NUM_ACTION)
            return action
        else:
            qs = self.model.predict(s.reshape(1,len(s)))[0]
            action = np.argmax(qs)
            is_greedy_index = np.where(qs == action)[0]
            if len(is_greedy_index) > 1:
                action = np.random.choice(is_greedy_index)
            return action

    def train(self,s,a,reward,s_dash,a_dash):
        # special reward for DQN
        if use_clip_reward:
            reward = 0
            if self.done:
                if reward > -200:
                    reward = +1
                else:
                    reward = -1

        self.buff.add(s,a,reward,s_dash,a_dash,self.done)
        batch = self.buff.getBatch(self.num_batch)
        s_batch =  np.asarray([e[0] for e in batch])
        a_batch =  np.asarray([e[1] for e in batch])
        r_batch =  np.asarray([e[2] for e in batch])
        s_dash_batch =  np.asarray([e[3] for e in batch])
        a_dash_batch =  np.asarray([e[4] for e in batch])
        done_batch =  np.asarray([e[5] for e in batch])
 
        Q_val_dash = np.max(self.model.predict(s_dash_batch), axis=1)
        targets = self.model.predict(s_batch)
        target = r_batch + GAMMA * Q_val_dash * (done_batch -1.) * -1. # means r when done else r + GAMMA * Q'
        for i, _a in enumerate(a_batch):
            targets[i,_a] = target[i]
        self.model.train_on_batch(s_batch, targets)

    def syncModel(self):
        self.targetModel.set_weights(self.model.get_weights())


    def calc_Z(self):
        x = np.linspace(self.min_list[0],self.max_list[0],meshgrid)
        y = np.linspace(self.min_list[1],self.max_list[1],meshgrid)
        X,Y = np.meshgrid(x,y)
        Z = np.array([[[agent.getQ(np.array([i,j]),k) for i in x] for j in y] for k in range(NUM_ACTION)])
        return X, Y, Z


class ReplayBuffer:

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def getBatch(self, batch_size):
        # Randomly sample batch_size examples
        if self.num_experiences < batch_size:
            return random.sample(self.buffer, self.num_experiences)
        else:
            return random.sample(self.buffer, batch_size)

    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, new_state,new_action,  done):
        experience = (state, action, reward, new_state, new_action, done)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0

class Result:

    def __init__(self):
        self.action_list = []
        self.reward_list = []
        self.time_list = []

    def draw_meshgrid(self, agent, draw_all_action):
        X,Y,Z = agent.calc_Z()

        if (draw_all_action):
            fig = plt.figure(figsize=(40,10))
            plt.title('episode: %4d,   epsilon: %.3f,   alpha: %.3f,   average_reward: %3f' %(epi, EPSILON, ALPHA, np.mean(self.reward_list)))

            ax1 = fig.add_subplot(141, projection='3d')
            plt.gca().invert_zaxis()
            ax1.plot_wireframe(X,Y,Z[0], rstride=1, cstride=1)

            ax2 = fig.add_subplot(142, projection='3d')
            plt.gca().invert_zaxis()
            ax2.plot_wireframe(X,Y,Z[1], rstride=1, cstride=1)
            
            ax3 = fig.add_subplot(143, projection='3d')
            plt.gca().invert_zaxis()
            ax3.plot_wireframe(X,Y,Z[2], rstride=1, cstride=1)


            ax4 = fig.add_subplot(144)
            sns.heatmap(np.argmax(Z, axis=0))
            plt.gca().invert_yaxis()
        else:
            fig = plt.figure(figsize=(10,10))

            ax1 = fig.add_subplot(111, projection='3d')
            ax1.set_xlabel('position')
            ax1.set_ylabel('speed')
            ax1.set_zlabel('V')

            ax1.set_title('episode: %4d,   epsilon: %.3f,   alpha: %.3f,   average_reward: %3d' %(epi, EPSILON, ALPHA, np.mean(self.reward_list)))
            plt.gca().invert_zaxis()
            ax1.plot_wireframe(X,Y,Z[2], rstride=1, cstride=1)

    def get_moving_ave(self, tap):
        d = np.ones(tap)/(1.0*tap)
        data = np.convolve(self.reward_list, d, 'same')
        return data
            

        



# Agentの選択
#agent = LinearFuncAgent(N=30, sigma=0.1,alpha_ini=0.2, alpha_last=0, eps_ini=0, eps_last=0)
#agent = TableAgent(N=30, alpha_ini=0.5, alpha_last=0, eps_ini=0, eps_last=0)
agent = DqnAgent(lr=1.E-4, h1=32, h2=16, TAU=5, num_memory = 10000, in_dim=NUM_STATE, out_dim=NUM_ACTION, eps_ini=0, eps_last=0, use_clip_reward=False)

reward_list = []
res = Result()  # 結果を格納するクラス

start_time = time.time()
for epi in range(int(num_episode)):
    
   
    res.action_result = np.zeros(NUM_ACTION)
    tmp = 0 # 報酬積算用
    count = 0

    if agent.__class__.__name__ == 'DqnAgent':
        if epi % agent.TAU == 0:
            agent.syncModel()
            print("targetNN sync!!")


    # greedy方策を徐々に確定的にしていく
    EPSILON = max(agent.eps_last, agent.eps_ini* (1- epi*1./num_episode))
    ALPHA = max(agent.alpha_last, agent.alpha_ini* (1- epi*1./num_episode))
 

    done = False
    # initialize s
    s = env.reset() # 環境をリセット
    agent.done = False
    # e-greedyによる行動選択   
    a = agent.select_action(s,EPSILON)

    while(agent.done==False):
        if render:
            env.render()

        # 行動aをとり、r, s'を観測
        s_dash, reward, done, info = env.step(a)
        agent.done = done
        # s'からe-greedyにより次の行動を決定
        a_dash = agent.select_action(s_dash,EPSILON)
        # 価値（パラメータ）の更新
        agent.train(s,a,reward,s_dash,a_dash)

        if agent.done:
            if count < 199:
                print("SUCEED!!")

        a = a_dash
        s = s_dash

        res.action_result[a_dash] +=1
        tmp += reward
        if (one_step_record) :
            X,Y,Z = agent.calc_Z()
            res.draw_meshgrid(agent, draw_all_action)
            plt.savefig(path + now + '/fig/mountaincar_Q_%04d_%03d.png' % (epi, count))
            plt.close()
        count +=1


    res.reward_list.append(tmp)
    res.time_list.append(time.time() - start_time)
    
    #print( agent.theta_list)
    print("epi: %d, eps: %.3f, alpha: %.3f, reward %d: " % (epi, EPSILON, ALPHA, tmp))
    print(res.action_result)

            
            

    if (fileOut & (epi %rec_interval == 0)) :
        X,Y,Z = agent.calc_Z()
        res.draw_meshgrid(agent, draw_all_action)
        plt.savefig(path + now + '/fig/mountaincar_Q_%04d.png' % (epi))
        plt.close()

env.close()

tap = 10
data = res.get_moving_ave(tap=10)
plt.plot(data)
plt.title('reward_list tap:%d' % tap)
plt.savefig(path + now + '/fig/reward_list.png')
plt.close()

plt.plot(res.time_list, data)
plt.title('reward_list tap:%d' % tap)
plt.savefig(path + now + '/fig/reward_list_time.png')
plt.close()

np.savez(path+now+  "/theta/res.npz", action_list=np.array(res.action_list),\
                    reward_list=np.array(res.reward_list),\
                    time_list=np.array(res.time_list))


