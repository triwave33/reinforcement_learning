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

# Ai GymのCartPoleを使用
#game = 'CartPole-v0'
game = 'MountainCar-v0'

if  game == 'CartPole-v0':
    env = gym.make('CartPole-v0')
    lows = np.array([-4.8, -5, -0.4, -5])
    highs = np.array([4.8, 5, 0.5, 5])
    num_state = 4
    num_action = 2
    select_num = 0
else:
    env = gym.make('MountainCar-v0')
    lows = env.observation_space.low
    highs = env.observation_space.high
    num_state = 2
    num_action = 3
    select_num =0

N =3 # N分割
min_list = env.observation_space.low  # 下限値 [-1.2 -0.07]
max_list = env.observation_space.high  # 上限値 [-0.6, 0.07]

s1_space = np.linspace(min_list[0], max_list[0], N)
s2_space = np.linspace(min_list[1], max_list[1], N)

b = (N**num_state)

num_x = num_state + num_action

c = np.random.rand(num_x, b)
c = np.zeros((num_x, b))
cnt =0
for i in s1_space:
    for j in s2_space:
        c[0,cnt] =i
        c[1,cnt] =j
        cnt+=1
c = np.tile(c,(1,3))
for i in range(num_action):
    c[2+i,(i*3**2):(i+1)*3**2] =1

b *= num_action


use_onehot_action = True 
print_log = False
M = 10
T = 200

ACTIONS = [0,1,2]
ALPHA = 0.1
ALPHA_INI = .5
ALPHA_LAST = .5
GAMMA = 0.9
EPS_INI = 0.2
EPS_LAST = 0.20
num_episode = 1000 # サンプリング数
render = 0 # 描写モード
ex_factor = 1.0 # epsilonがゼロになったあとも学習を続けるパラメータ
use_potential_reward = True


# 基底関数
def rbf(s,a, cb, sigma):
    if use_onehot_action:
        a = one_hot(a)
    x = np.hstack([s,a])
    return np.exp(-np.square(LA.norm(x-cb))/(2*sigma**2)) 

num_rbf = 1
if use_onehot_action:
    num_x = num_state + num_action
else:
    num_x = num_state +1

sigma=1
theta = np.random.rand(b) # 初期化
#c = np.zeros((num_x,b))
c = np.random.rand(num_x,b)
for i in range(num_state):
    c[i,:] = np.linspace(lows[i],highs[i],b)


# Q
def Q(s, a, theta,c):
    res =0
    for i in range(b):
        cb = c[:,i]
        res += theta[i] * rbf(s,a,cb,sigma)
    return res

def one_hot(a):
    assert a <num_action
    array = np.zeros(num_action)
    array[a] = 1
    return array


# epsilon-greedyのための確率分布を初期化
p = np.random.rand(num_action)
p = p / np.sum(p, axis=0) # 正規化



def select_action(s,theta,c, eps):
    # e-greedyによる行動選択   
    if np.random.rand() > eps:
        qs = [Q(s, i, theta, c)\
            for i in range(num_action)]
        action = np.argmax(qs)
        if print_log:
            print qs
            print action
    else:
        action = np.random.randint(select_num ,num_action)
    return action
   

upsampling = 1

for up in range(upsampling):
    for trial in range(1):
        #for epi in tqdm(range(int(num_episode*ex_factor))):
        reward_list = []
        min_pos = min_list[0]
        max_pos = max_list[1]

        for epi in range(int(num_episode*ex_factor)):
            # greedy方策を徐々に確定的にしていく
            EPSILON = max(EPS_LAST, EPS_INI* (1- epi*1./num_episode))
            ALPHA = max(ALPHA_LAST, ALPHA_INI* (1- epi*1./num_episode))
            

            visit_list = []
            action_result = np.zeros(num_action)
            # エピソードを終端までプレイ
            for i in range(M):
                 # sの初期化
                done = False
                s = env.reset() # 環境をリセット
            
                # e-greedyによる行動選択   
                a = select_action(s, theta, c, EPSILON)
                tmp = 0 # 報酬積算用

                count = 0
                while(done==False):
 
                    if render:
                        env.render()
        
        
                    # 行動aをとり、r, s'を観測
                    s_dash, reward, done, info = env.step(a)
                    if use_potential_reward:
                        reward = s_dash[0]**2
                    a_dash = select_action(s_dash, theta, c, EPSILON)
                    action_result[a_dash] +=1

                    visit_list.append([s,a,reward,s_dash,a_dash])

                    s = s_dash
                    a = a_dash

                    
                    tmp += reward
                    count +=1

                if count <200:
                    print("SUCEED")
        

                reward_list.append(tmp)
            memory = np.array(visit_list)
            
            sAll = memory[:,0]
            aAll = memory[:,1]
            rAll = np.array(memory[:,2])
            ssAll = memory[:,3]
            aaAll  =memory[:,4]

            print("now calculating")

            meanPhi = [np.mean([rbf(ssAll[i], aaAll[i], c[:,j], sigma) \
                    for i in range(len(aaAll))]) for j in range(b)]
            meanPhi = np.array(meanPhi)

            Phi = [[rbf(ssAll[i], aaAll[i], c[:,j], sigma) for i in range(len(aaAll))] for j in range(b)]
            Phi = np.array(Phi).T
            X = Phi - GAMMA * meanPhi

            theta_new = LA.inv(X.T.dot(X)).dot(X.T).dot(rAll)

            theta = theta + (ALPHA* (theta_new - theta))
            print( theta)
            print("epi: %d, eps: %f, reward %f: " % (epi, EPSILON, np.mean(reward_list)))
            print(action_result)

            
            
            

#            if ((epi %500 == 0) | ((epi <500) & (epi %50==0))) :
#                x = np.linspace(min_list[0],max_list[0],N)[1:-1]
#                y = np.linspace(min_list[1],max_list[1],N)[1:-1]
#                X,Y = np.meshgrid(x,y)
#
#                if (False):
#                    fig = plt.figure(figsize=(40,10))
#                    plt.title('episode: %4d,   epsilon: %.3f,   alpha: %.3f,   average_reward: %3d' %(epi, EPSILON, ALPHA, np.mean(reward_list)))
#
#                    ax1 = fig.add_subplot(141, projection='3d')
#                    ax1.set_zlim(-40,0)
#                    plt.gca().invert_zaxis()
#                    ax1.plot_wireframe(X,Y,Q[0,:,:], rstride=1, cstride=1)
#
#                    ax2 = fig.add_subplot(142, projection='3d')
#                    ax2.set_zlim(-40,0)
#                    plt.gca().invert_zaxis()
#                    ax2.plot_wireframe(X,Y,Q[1,:,:], rstride=2, cstride=2, linewidth=.1)
#
#                    ax3 = fig.add_subplot(143, projection='3d')
#                    ax3.set_zlim(-40,0)
#                    plt.gca().invert_zaxis()
#                    ax3.plot_wireframe(X,Y,Q[2,:,:], rstride=2, cstride=2)
#
#                    ax4 = fig.add_subplot(144)
#                    sns.heatmap(np.argmax(Q, axis=0))
#                else:
#                    fig = plt.figure(figsize=(10,10))
#
#                    ax1 = fig.add_subplot(111, projection='3d')
#                    ax1.set_xlabel('position')
#                    ax1.set_ylabel('speed')
#                    ax1.set_zlabel('V')
#
#                    ax1.set_title('episode: %4d,   epsilon: %.3f,   alpha: %.3f,   average_reward: %3d' %(epi, EPSILON, ALPHA, np.mean(reward_list)))
#                    #ax1.set_zlim(-40,0)
#                    #plt.gca().invert_zaxis()
#                    ax1.plot_wireframe(X,Y,np.mean(Q,axis=0)[1:-1,1:-1], rstride=1, cstride=1)
#                    
#
#
#
#
#
#
#                plt.savefig('mountaincar_Q_%d_%04d.png' % (up, epi))
#                plt.close()
#
#                reward_list = []
#
#
#     
#            print("N: %d, epi: %d, eps#: %.3f, reward: %3d, min_pos: %d, max_pos: %d" % (N, epi, EPSILON, tmp, min_pos, max_pos))
#            Q_10trial[trial, epi] = tmp
#            reward_array[epi] = tmp
#            env.close()
#    N *= 2
#    Q = Q.repeat(2,axis=1).repeat(2,axis=2)
#    visit_array = visit_array.repeat(2,axis=1).repeat(2,axis=2)
#
#epi=0
#while(True):
#    done = False
#    observation = env.reset() # 環境をリセット
#    # 状態を離散化
#    s = [int(np.digitize(observation[i], np.linspace(min_list[i],\
#            max_list[i], N-1))) for i in range(num_state)]
#    # e-greedyによる行動選択   
#    a = select_action(Q, s, EPSILON)
#    tmp = 0
#    while(done == False):
#        # 行動aをとり、r, s'を観測
#        observation, reward, done, info = env.step(a)
#        tmp += reward
#        if render:
#            env.render()
#
#        # 状態を離散化
#        s_dash = [int(np.digitize(observation[i], np.linspace(min_list[i], \
#            max_list[i], N-1))) for i in range(num_state)]
#
#        # e-greedyによる行動選択   
#        a = select_action(Q, s_dash, EPSILON)
#    print("epi: %d, reward: %3d" % (epi, tmp))
#
#plt.plot(reward_array)
#plt.show()
