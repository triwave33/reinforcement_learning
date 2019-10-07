### -*-coding:utf-8-*-
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

############### 
#true_online_sarsa_wx_mountaincar_2.py
# 1との違い
#1: 入力空間がSxA、出力がスカラ
#2: 入力空間がS、出力がA


ACTIONS = [0,1,2]
ALPHA =1.0E-8
ALPHA_INI = 2.E-2
ALPHA_LAST = 0.0E-2
GAMMA = 0.99
EPS_INI = 0.0
EPS_LAST = 0
LAMBDA = 0.0
render = 0 # 描写モード
ex_factor = 1.0 # epsilonがゼロになったあとも学習を続けるパラメータ
sigma=0.1
print(EPS_INI)
use_potential_reward = False # 位置に応じた報酬
use_velosity_reward = False # 速度に応じた報酬
use_binary_action = False # 左・右のみのアクション
num_episode = 1601

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

# 定数項の決定
c = np.random.rand(num_x, b) # ランダムの場合
c = np.zeros((num_x, b)) # オール0の場合
cnt =0
for i in s1_space:
    for j in s2_space:
        c[0,cnt] =i
        c[1,cnt] =j
        cnt+=1

c_list = [np.copy(c),np.copy(c),np.copy(c)] 

print_log = False

# 基底関数
norm_factor = np.array([1.2, 0.07])
norm_factor = norm_factor.reshape(len(norm_factor),1)

def rbfs(s, c, sigma):
    x = s
    x = x.reshape(len(x),1)
    return np.exp(-np.square(LA.norm((c-x)/norm_factor, axis=0))/(2*sigma**2)) 

def rbf_taihi(s,a, cb, sigma, num_action):
    a = one_hot(a, num_action)
    x = np.hstack([s,a])
    return np.exp(-np.square(LA.norm((x-cb)/norm_factor))/(2*sigma**2)) 


np.savez(path + now + '/theta/constants',c=c, b=b, sigma=sigma, num_action=num_action, norm_factor=norm_factor)

# Q
def Q(s, theta,c,sigma):
    return rbfs(s,c,sigma).dot(theta)



# Q
def Q_for_meshgrid(s0,s1, theta,c,sigma):
    s = np.hstack([s0,s1])
    x = rbfs(s,c,sigma)
    return theta.dot(x)


def select_action(s,theta_list, c, eps, num_action,sigma):
    # e-greedyによる行動選択   
    if np.random.rand() > eps:
        qs = [Q(s, theta_list[i], c, sigma)\
            for i in range(num_action)]
        action = np.argmax(qs)
        if print_log:
            print qs
            print action
    else:
        action = np.random.randint(select_num ,num_action)
    return action
   

upsampling = 1

#for epi in tqdm(range(int(num_episode*ex_factor))):
s_list = []
a_list = []
min_pos = min_list[0]
max_pos = max_list[1]

result = []
for n in range(5): # 初期化
    theta_list = [np.zeros(b), np.zeros(b), np.zeros(b)] # 初期化
    e = np.zeros(b)
    reward_list = []

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
        
        # e-greedyによる行動選択   
        a = select_action(s, theta_list, c, EPSILON, num_action, sigma)
    
        tmp = 0 # 報酬積算用
        
        #x = np.array([rbf(s,a,c[:,i],sigma, num_action) for i in range(c.shape[1])])
        x = rbfs(s,c,sigma)
    
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
    
    
            a_dash = select_action(s_dash,theta_list, c, EPSILON,num_action, sigma)
            
            x_dash = rbfs(s_dash,c,sigma)
    
            Q_val = theta_list[a].dot(x)
            Q_val_dash = theta_list[a_dash].dot(x_dash)
    
            DELTA = reward + GAMMA*Q_val_dash - Q_val
    
            z = GAMMA * LAMBDA * z + (1- ALPHA * GAMMA * LAMBDA * ((z.T).dot(x)))*x
    
            theta_list[a] = theta_list[a] + ALPHA*(DELTA + Q_val - Q_val_old)*z - ALPHA*(Q_val - Q_val_old)*x
            #theta_list[a] = theta_list[a_dash] + ALPHA*(DELTA) *x
    
            Q_val_old = Q_val_dash
    
            x = x_dash
            a = a_dash
            s = s_dash
    
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
        print("epi: %d, eps: %f, alpha: %f, TDerr: %f  reward %f: " % (epi, EPSILON, ALPHA, DELTA, tmp))
        print(action_result)
    
                
                
                
    
        if ((epi %grid_interval == 0) | ((epi <2000) & (epi %500==0))) :
            x = np.linspace(min_list[0],max_list[0],meshgrid)
            y = np.linspace(min_list[1],max_list[1],meshgrid)
            X,Y = np.meshgrid(x,y)
            Z = np.array([[[Q_for_meshgrid(i,j,theta_list[k],c,sigma) for i in x] for j in y] for k in range(num_action)])
    
            if (False):
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
                
    
    
    
            np.save(path + now +  '/theta/mountaincar_Q_theta_%04d.npy' % (epi), np.array(theta_list))
            plt.savefig(path + now + '/fig/mountaincar_Q_%d_%04d.png' % (n, epi))
            plt.close()
    
    result.append(reward_list)
    env.close()

plt.plot(reward_list)
plt.savefig(path + now + '/fig/reward_list.png')
plt.close()
np.save(path + now + '/theta/result.npy' , np.array(result))
np.save(path + now + '/theta/reward_list' , np.array(reward_list))
np.save(path + now + '/theta/s_list' , np.array(s_list))
np.save(path + now + '/theta/a_list' , np.array(a_list))
