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
from keras.optimizers import Adam, SGD, RMSprop
from keras import backend as K
import tensorflow as tf
import  gym
import math
import seaborn as sns
import os
import datetime
import shutil
import sys
from ReplayBuffer import ReplayBuffer


ACTIONS = [0,1,2]
GAMMA = 0.99
EPS_INI = 0.0
EPS_LAST = 0.0
LAMBDA = 0.6
render = 0 # 描写モード
TAU =5
ex_factor = 2.0 # epsilonがゼロになったあとも学習を続けるパラメータ
use_potential_reward = False # 位置に応じた報酬
use_velosity_reward = False # 速度に応じた報酬
use_binary_action = False # 左・右のみのアクション
num_episode = 801
num_memory = 10000
num_batch = 16
learning_rate = 1E-4
h1 = 32
h2 = 16
dqn = False
if dqn:
    TAU = TAU
    num_memory = num_memory
    loss = huberloss
else:
    TAU = 1
    num_memory = 10000
    loss = 'mse'
N =30 # N分割
meshgrid = 25
grid_interval = 25

# Ai GymのCartPoleを使用
game = 'CartPole-v0'
game = 'MountainCar-v0'
env = gym.make(game)
lows = env.observation_space.low
highs = env.observation_space.high
if game == 'CartPole-v0':
    num_state = 4
    num_action = 2
elif game == 'MountainCar-v0':
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

def huberloss(y_true, y_pred):
    err = y_true - y_pred
    cond = K.abs(err) < 1.0
    L2 = 0.5 * K.square(err)
    L1 = (K.abs(err) - 0.5)
    loss = tf.where(cond, L2, L1)  # Keras does not cover where function in tensorflow :-(
    return K.mean(loss)


class NN():
    def __init__(self, lr, h1, h2, input_dim, output_dim):
        self.lr = lr
        self.h1 = h1
        self.h2 = h2
        self.input_dim = (input_dim,)
        self.output_dim = output_dim
        self.model = self.build_model()

        #optimizer = Adam(self.lr)
        optimizer = RMSprop(self.lr)


        self.model.compile(loss= loss, optimizer=optimizer, metrics = ['accuracy'])

    

    def build_model(self):
        model = Sequential()
        model.add(Dense(self.h1, input_shape = self.input_dim, activation='relu'))
        model.add(Dense(self.h2, activation='relu'))
        model.add(Dense(self.output_dim, activation='linear',init='zero'))

        model.summary()

        return model

    def train(self, x, y, epochs, batch_size=1):
        loss = self.model.train_on_batch(x, y)
        
        

def select_action(s, model, eps, num_action):
    # e-greedyによる行動選択   
    if np.random.rand() > eps:
        s = s.reshape(1,len(s))
        qs = model.predict(s)
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


def main(TAU, num_memory, loss, use_clip_reward, num=0):
    #for epi in tqdm(range(int(num_episode*ex_factor))):
    reward_list = []
    s_list = []
    a_list = []
    min_pos = min_list[0]
    max_pos = max_list[1]
    
    now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    path = './'
    #path = '/volumes/data/dataset/ai_gym/'
    os.mkdir(path +now)
    os.mkdir(path +now + '/fig')
    os.mkdir(path +now + '/theta')
    myfile = os.path.dirname(os.path.abspath(__file__)) + '/true_online_sarsa_NN_mountaincar.py'
    shutil.copyfile(myfile, path +now + '/theta.setting.py')

    
    agent = NN(lr=learning_rate, h1=h1, h2=h2, input_dim=num_state, output_dim=num_action)
    #agent.model.load_weights('weights.h5')
    targetNN = NN(lr=learning_rate, h1=h1, h2=h2, input_dim=num_state, output_dim=num_action)
    
    buff = ReplayBuffer(num_memory)
    
    
    targetNN.model.set_weights(agent.model.get_weights())
    episode_reward_list =[]
    succeed_count = 0
    result = []
    
    count = 0
    for epi in range(int(num_episode*ex_factor)):
        # greedy方策を徐々に確定的にしていく
        EPSILON = max(EPS_LAST, EPS_INI* (1- epi*1./num_episode))
        #EPSILON = 0.1 + 0.9 /(1.+epi)
        #EPSILON = 0.1 + 0.9 /(1.+.1*epi)
        visit_list = []
        action_result = np.zeros(num_action)
        # エピソードを終端までプレイ
         # sの初期化
        done = False
    
        # initialize s
        s = env.reset() # 環境をリセット
        #if epi ==0:
        #    s = np.array([-0.44982587,0.])  # 再現性をチェックする時用のシード
    
        
        # e-greedyによる行動選択   
        a = select_action(s, agent.model, EPSILON, num_action)
    
    
        tmp = 0 # 報酬積算用
        
        #x = np.array([rbf(s,a,c[:,i],sigma, num_action) for i in range(c.shape[1])])
    
        step_count = 0
        Q_val_old = 0
        z = np.zeros(b)
    
        s_list_episode = []
        a_list_episode = []
        step_count_list = []
    
        episode_reward = 0  # 結果表示用の報酬
    
        best_pos = -0.5
    
        while(done==False):
            if render:
                env.render()
    
            
    
            s_dash, reward, done, info = env.step(a)
    
            if best_pos < s_dash[0]:
                best_pos = s_dash[0]
    
            if use_clip_reward:
                reward = 0
                if done:
                    if game=='CartPole-v0':
                        if step_count > 198:
                            reward = 1
                            print("succeed")
                            succeed_count += 1
                        else:
                            reward = -1
                    elif game == 'MountainCar-v0':
                        if step_count < 199:
                            reward = 1
                            print("succeed")
                            succeed_count += 1
                        else:
                            reward = -1
    
            a_dash = select_action(s_dash,agent.model,  EPSILON,num_action)
            visit_list.append([s,a,reward,s_dash,a_dash])
    
            buff.add(s,a,reward,s_dash,a_dash, done)
    
    
            
            batch = buff.getBatch(num_batch)
            s_batch =  np.asarray([e[0] for e in batch])
            a_batch =  np.asarray([e[1] for e in batch])
            r_batch =  np.asarray([e[2] for e in batch])
            s_dash_batch =  np.asarray([e[3] for e in batch])
            a_dash_batch =  np.asarray([e[4] for e in batch])
            done_batch =  np.asarray([e[5] for e in batch])
    
            #for i, (s,a,r,ss,aa) in enumerate(mini_batch):
            #    inputs[i,:] = s
            #    Q_val_dash = np.max(targetNN.model.predict(ss.reshape((1,num_state)))[0])
            #    targets[i] = agent.model.predict(s.reshape((1,num_state)))[0]
    
            #    if done:
            #        target = r
            #    else:
            #        target = r + GAMMA * Q_val_dash
            #    targets[i,a] = target
            
            Q_val_dash = np.max(targetNN.model.predict(s_dash_batch), axis=1)
            targets = agent.model.predict(s_batch)
            target = r_batch + GAMMA * Q_val_dash * (done_batch -1.) * -1. # means r when done else r + GAMMA * Q'
            for i, _a in enumerate(a_batch):
                targets[i,_a] = target[i]
    
    
            loss = agent.model.train_on_batch(s_batch ,targets)
    
            a_list_episode.append(a)
            s_list_episode.append(s)
            a = a_dash
            s = s_dash
    
            action_result[a_dash] +=1
    
            tmp += reward
            step_count +=1
            episode_reward += 1
    
        if epi %TAU ==0:
            targetNN.model.set_weights(agent.model.get_weights())        
            print("weight updated")
        step_count_list.append(step_count)
        reward_list.append(tmp)
        s_list.append(s_list_episode)
        a_list.append(a_list_episode)
        memory = np.array(visit_list)
        
        episode_reward_list.append(episode_reward)
        #print( theta_list)
        print("epi:%d, eps:%.2f, t:%d, x:%.2f, #:%d, r:%d, l: %.1e" % (epi, EPSILON,step_count, best_pos, succeed_count,  tmp, loss[0]))
        #print(action_result)
    
        count += 1
    
                
                
                
    
        if ((epi %grid_interval == 0) | ((epi <2000) & (epi %500==0))) :
            x = np.linspace(min_list[0],max_list[0],meshgrid)
            y = np.linspace(min_list[1],max_list[1],meshgrid)
            X,Y = np.meshgrid(x,y)
            if game=='CartPole-v0':
                Z = np.array([[agent.model.predict(np.hstack([i,j,0,0]).\
                reshape(1,num_state))[0] for i in x] for j in y])
            elif game== 'MountainCar-v0':
                Z = np.array([[agent.model.predict(np.hstack([i,j]).\
                reshape(1,num_state))[0] for i in x] for j in y])
    
    
            if (False):
                fig = plt.figure(figsize=(40,10))
                plt.title('episode: %4d,   epsilon: %.3f,   average_reward: %3d' %(epi, EPSILON,  np.mean(reward_list)))
    
                for i in range(num_action):
                    ax = fig.add_subplot(1,num_action+1,i+1, projection='3d')
                    plt.gca().invert_zaxis()
                    ax.plot_wireframe(X,Y,Z[:,:,i], rstride=1, cstride=1)
    
    
                ax1 = fig.add_subplot(141, projection='3d')
                plt.gca().invert_zaxis()
                ax1.plot_wireframe(X,Y,Z[:,:,0], rstride=1, cstride=1)
    
                ax2 = fig.add_subplot(142, projection='3d')
                plt.gca().invert_zaxis()
                ax2.plot_wireframe(X,Y,Z[:,:,1], rstride=1, cstride=1)
                
                if use_binary_action != True:
                    ax3 = fig.add_subplot(143, projection='3d')
                    plt.gca().invert_zaxis()
                    ax3.plot_wireframe(X,Y,Z[:,:,2], rstride=1, cstride=1)
    
    
                ax4 = fig.add_subplot(144)
                sns.heatmap(np.argmax(Z, axis=2))
                plt.gca().invert_yaxis()
            else:
                fig = plt.figure(figsize=(10,10))
    
                ax1 = fig.add_subplot(111, projection='3d')
                ax1.set_xlabel('position')
                ax1.set_ylabel('speed')
                ax1.set_zlabel('V')
    
                ax1.set_title('episode: %4d,   epsilon: %.3f, average_reward: %3d' %(epi, EPSILON, np.mean(reward_list)))
                #ax1.set_zlim(-40,0)
                plt.gca().invert_zaxis()
                ax1.plot_wireframe(X,Y,Z[:,:,2], rstride=1, cstride=1)
                
    
    
    
            plt.savefig(path + now + '/fig/mountaincar_Q_%d_%04d.png' % (num, epi))
            plt.close()
    
    
        env.close()
    
    #plt.plot(reward_list)
    #plt.savefig(path + now + '/fig/reward_list.png')
    #plt.close()
    np.save(path + now + '/theta/reward_list' , np.array(reward_list))
    np.save(path + now + '/theta/s_list' , np.array(s_list))
    np.save(path + now + '/theta/a_list' , np.array(a_list))
    agent.model.save_weights(path + now + '/theta/weights.h5')
    agent.model.save(path + now + '/theta/model.h5')

    return episode_reward_list



# 1 pure NN
res1 = []
for i in range(5):
    res1.append(main(TAU=1, num_memory=1, loss='mse', use_clip_reward=False, num=i))


# 2 pure NN + cr + huberloss
res2 = []
for i in range(5):
    res2.append(main(TAU=1, num_memory=1, loss='huberloss', use_clip_reward=True, num=i))


## 3 pure NN + cr+ huberloss + er
#res3 = []
#for i in range(5):
#    res3.append(main(TAU=1, num_memory=100, loss='huberloss', use_clip_reward=True, num=i))
#
#
## 4 pure NN + cr + huberloss + er
#res4 = []
#for i in range(5):
#    res4.append(main(TAU=1, num_memory=10000, loss='huberloss', use_clip_reward=True, num=i))
#
#
## 5 pure NN + cr + huberloss + er + targetNN
#res5 = []
#for i in range(5):
#    res5.append(main(TAU=5, num_memory=10000, loss='huberloss', use_clip_reward=True, num=i))
#
## 6 pure NN + cr+ huberloss + er + targetNN
#res6 = []
#for i in range(5):
#    res6.append(main(TAU=25, num_memory=10000, loss='huberloss', use_clip_reward=True, num=i))
#
#
# 7 pure NN + clip_reward
res7 = []
for i in range(5):
    res2.append(main(TAU=1, num_memory=1, loss='mse', use_clip_reward=True, num=i))


results = [res1,res2,res3,res4,res5,res6]
plt.savefig(path +  'dqn_result.png')

labels = ['NN', 'NN+HL', 'NN+HL+ER100', 'NN+HL + ER10000',\
        'NN+HL+ER10000+TN5', 'NN+HL+ER10000+TN25']
for i,r in enumerate(resuls):
    np.save(path + 'results%s.npy' % labels[i], results[i])
    f = np.mean(r, axis=0) * -1.
    plt.plot(f, label= labels[i])
