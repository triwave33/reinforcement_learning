### -*-coding:utf-8-*-
import numpy as np
import scipy.linalg as LA
from itertools import product
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tqdm import tqdm
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, SGD
import  gym
import math
import seaborn as sns
import os
import datetime
import shutil
import sys


ACTIONS = [0,1,2]
ALPHA =1.0E-8
ALPHA_INI = 2.E-1
ALPHA_LAST = 0.0E-2
GAMMA = 0.98
EPS_INI = 0.0
EPS_LAST = 0
LAMBDA = 0.6
render = 0 # 描写モード
ex_factor = 1.0 # epsilonがゼロになったあとも学習を続けるパラメータ
sigma=0.1
print(EPS_INI)
use_potential_reward = False # 位置に応じた報酬
use_velosity_reward = False # 速度に応じた報酬
use_binary_action = False # 左・右のみのアクション
num_episode = 301

now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')


path = './'
#path = '/volumes/data/dataset/ai_gym/'
os.mkdir(path +now)
os.mkdir(path +now + '/fig')
os.mkdir(path +now + '/theta')
myfile = os.path.dirname(os.path.abspath(__file__)) + '/true_online_sarsa_wx_mountaincar.py'
shutil.copyfile(myfile, path +now + '/theta.setting.py')
N =30 # N分割
meshgrid = 25
grid_interval = 25

# Ai GymのCartPoleを使用
#game = 'CartPole-v0'
game = 'MountainCar-v0'
env = gym.make('MountainCar-v0')
lows = env.observation_space.low
highs = env.observation_space.high
num_state = 2
num_action = 3
if use_binary_action:
    num_action = 2
select_num =0

min_list = env.observation_space.low  # 下限値 [-1.2 -0.07]
max_list = env.observation_space.high  # 上限値 [-0.6, 0.07]
s1_space = np.linspace(min_list[0], max_list[0], N)
s2_space = np.linspace(min_list[1], max_list[1], N)
b = (N**num_state)
num_x = num_state # 入力空間に行動は含まない

print_log = False


class NN():
    def __init__(self, lr, h1, h2, input_dim, output_dim):
        self.lr = lr
        self.h1 = h1
        self.h2 = h2
        self.input_dim = (input_dim,)
        self.output_dim = output_dim
        self.model = self.build_model()

        optimizer = Adam(self.lr, 0.5)


        self.model.compile(loss= 'mse', optimizer=optimizer, metrics = ['accuracy'])

    

    def build_model(self):
        model = Sequential()
        model.add(Dense(self.h1, input_shape = self.input_dim))
        model.add(LeakyReLU(alpha=.2))
        model.add(Dense(self.h2))
        model.add(LeakyReLU(alpha=.2))
        model.add(Dense(self.output_dim))

        model.summary()

        return model

    def train(self, x, y, epochs, batch_size=1):
        loss = self.model.train_on_batch(x, y)
        
        

def select_action(s, model, eps, num_action):
    # e-greedyによる行動選択   
    if np.random.rand() > eps:
        qs = [model.predict(np.hstack([s[0],one_hot(i, num_action)]).reshape(1,5)) for i in range(num_action)]
        action = np.argmax(qs)
        if print_log:
            print qs
            print action
    else:
        action = np.random.randint(select_num ,num_action)
    return action

def one_hot(a, num_action):
    ret = np.zeros(num_action)
    ret[a] = 1
    return ret
   

upsampling = 1

#for epi in tqdm(range(int(num_episode*ex_factor))):
reward_list = []
s_list = []
a_list = []
min_pos = min_list[0]
max_pos = max_list[1]

agent = NN(lr=1.E-4, h1=50, h2=20, input_dim=5, output_dim=1)


for epi in range(int(num_episode*ex_factor)):
    # greedy方策を徐々に確定的にしていく
    EPSILON = max(EPS_LAST, EPS_INI* (1- epi*1./num_episode))
    ALPHA = max(ALPHA_LAST, ALPHA_INI* (1- epi*1./num_episode))
    visit_list = []
    action_result = np.zeros(num_action)
    # エピソードを終端までプレイ
     # sの初期化
    done = False

    # initialize s
    s = env.reset() # 環境をリセット
    if epi ==0:
        s = np.array([-0.44982587,0.])  # 再現性をチェックする時用のシード

    s = s.reshape(1,len(s))
    
    # e-greedyによる行動選択   
    a = select_action(s, agent.model, EPSILON, num_action)

    x = np.hstack([s[0],one_hot(a,num_action)]).reshape(1,5)

    tmp = 0 # 報酬積算用
    
    #x = np.array([rbf(s,a,c[:,i],sigma, num_action) for i in range(c.shape[1])])

    count = 0
    Q_val_old = 0
    z = np.zeros(b)

    s_list_episode = []
    a_list_episode = []

    while(done==False):
        if render:
            env.render()

        s_list_episode.append(s)
        a_list_episode.append(a)
        

        # 行動aをとり、r, s'を観測
        if use_binary_action:
            s_dash, reward, done, info = env.step(a*2)
        else:
            s_dash, reward, done, info = env.step(a)

        s_dash = s_dash.reshape(1,len(s_dash))


        if use_potential_reward:
            reward += s_dash[0]**2 
            if s_dash[0] > 0:
                reward *=2

        if use_velosity_reward:
            reward += s_dash[1]**2

        if done:
            if count < 199:
                #reward += 100
                print("succeed")


        a_dash = select_action(s_dash,agent.model,  EPSILON,num_action)
        x_dash = np.hstack([s_dash[0],one_hot(a_dash,num_action)]).reshape(1,5)

        

        Q_val = agent.model.predict(x)
        Q_val_dash = agent.model.predict(x_dash)

        loss = agent.model.train_on_batch(x, Q_val_dash)


        a = a_dash
        s = s_dash
        x = x_dash

        action_result[a_dash] +=1
        visit_list.append([s,a,reward,s_dash,a_dash])



        tmp += reward
        count +=1

    if count <200:
        print("SUCEED")

    reward_list.append(tmp)
    s_list.append(s_list_episode)
    a_list.append(a_list_episode)
    memory = np.array(visit_list)
    
    #print( theta_list)
    print("epi: %d, eps: %f, reward: %f, loss: %e" % (epi, EPSILON, tmp, loss[0]))
    print(action_result)

            
            
            

    if ((epi %grid_interval == 0) | ((epi <2000) & (epi %500==0))) :
        x = np.linspace(min_list[0],max_list[0],meshgrid)
        y = np.linspace(min_list[1],max_list[1],meshgrid)
        X,Y = np.meshgrid(x,y)
        Z = np.array([[[agent.model.predict(np.hstack([i,j,one_hot(k,num_action)]).reshape(1,5))[0][0] for i in x] for j in y] for k in range(num_action)])

        if (True):
            fig = plt.figure(figsize=(40,10))
            plt.title('episode: %4d,   epsilon: %.3f,   alpha: %.3f,   average_reward: %3d' %(epi, EPSILON, ALPHA, np.mean(reward_list)))

            ax1 = fig.add_subplot(141, projection='3d')
            plt.gca().invert_zaxis()
            ax1.plot_wireframe(X,Y,Z[0], rstride=1, cstride=1)

            ax2 = fig.add_subplot(142, projection='3d')
            plt.gca().invert_zaxis()
            ax2.plot_wireframe(X,Y,Z[1], rstride=1, cstride=1)
            
            if use_binary_action != True:
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

            ax1.set_title('episode: %4d,   epsilon: %.3f,   alpha: %.3f,   average_reward: %3d' %(epi, EPSILON, ALPHA, np.mean(reward_list)))
            #ax1.set_zlim(-40,0)
            plt.gca().invert_zaxis()
            ax1.plot_wireframe(X,Y,Z[2], rstride=1, cstride=1)
            



        plt.savefig(path + now + '/fig/mountaincar_Q_%04d.png' % (epi))
        plt.close()


    env.close()

plt.plot(reward_list)
plt.savefig(path + now + '/fig/reward_list.png')
plt.close()
np.save(path + now + '/theta/reward_list' , np.array(reward_list))
np.save(path + now + '/theta/s_list' , np.array(s_list))
np.save(path + now + '/theta/a_list' , np.array(a_list))



