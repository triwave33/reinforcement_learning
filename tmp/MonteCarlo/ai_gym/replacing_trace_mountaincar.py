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

ACTIONS = [0,1,2]
ALPHA =1.0E-8
ALPHA_INI = 1.0E-4
ALPHA_LAST = 0
GAMMA = 0.9
EPS_INI = 0.8
EPS_LAST = 0
LAMBDA = 0.0
render = 0 # 描写モード
ex_factor = 1.0 # epsilonがゼロになったあとも学習を続けるパラメータ
sigma=0.6
use_potential_reward = False
num_episode = 1000

N =3 # N分割


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

min_list = env.observation_space.low  # 下限値 [-1.2 -0.07]
max_list = env.observation_space.high  # 上限値 [-0.6, 0.07]
s1_space = np.linspace(min_list[0], max_list[0], N)
s2_space = np.linspace(min_list[1], max_list[1], N)

b = (N**num_state)
num_x = num_state + num_action

# 定数項の決定
c = np.random.rand(num_x, b)
c = np.zeros((num_x, b))
cnt =0
for i in s1_space:
    for j in s2_space:
        c[0,cnt] =i
        c[1,cnt] =j
        cnt+=1
c = np.tile(c,(1,num_action))
for i in range(num_action):
    c[2+i,(i*num_action**2):(i+1)*num_action**2] =1
b *= num_action


print_log = False

# 基底関数
norm_factor = np.array([1.2, 0.07, 1, 1, 1])
def rbf(s,a, cb, sigma):
    a = one_hot(a)
    x = np.hstack([s,a])
    return np.exp(-np.square(LA.norm((x-cb)/norm_factor))/(2*sigma**2)) 

# 初期化
theta = np.random.rand(b) # 初期化
e = np.zeros(b)

# Q
def Q(s, a, theta,c):
    res =0
    for i in range(b):
        cb = c[:,i]
        res += theta[i] * rbf(s,a,cb,sigma)
    return res

# Q
def Q_for_meshgrid(s0,s1, a, theta,c):
    res =0
    s = np.hstack([s0,s1])
    for i in range(b):
        cb = c[:,i]
        res += theta[i] * rbf(s,a,cb,sigma)
    return res


def one_hot(a):
    assert a <num_action
    array = np.zeros(num_action)
    array[a] = 1
    return array

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
        e = GAMMA * LAMBDA * e
        
        #e[:] = 0
        #print(e)
        e[(a*num_action**2):((a+1)*num_action**2)] = 1
        #print(e)

        # 行動aをとり、r, s'を観測
        s_dash, reward, done, info = env.step(a)

        if use_potential_reward:
            reward = s_dash[0]**2
            if s[0]>0:
                reward*=2

        Q_val = Q(s,a,theta,c)

        DELTA = reward - Q_val
        a_dash = select_action(s_dash, theta, c, EPSILON)

        action_result[a_dash] +=1
        visit_list.append([s,a,reward,s_dash,a_dash])

        Q_dash_val = Q(s_dash, a_dash, theta, c)

        DELTA += GAMMA * Q_dash_val
        #print("Q") #print(Q_val) #print("ALPHA") #print(ALPHA)
        #print("DELTA")
        #print(DELTA)
        #theta += ALPHA * DELTA * e

        rfs = np.array([rbf(s,a,c[:,i],sigma) for i in range(c.shape[1])])
        theta -= ALPHA * DELTA * theta

        s = s_dash
        a = a_dash
        tmp += reward
        count +=1

    if count <200:
        print("SUCEED")

    reward_list.append(tmp)
    memory = np.array(visit_list)
    
    print( theta)
    print("epi: %d, eps: %f, alpha: %f, TDerr: %f  reward %f: " % (epi, EPSILON, ALPHA, DELTA, np.mean(reward_list)))
    print(action_result)

            
            
            

    if ((epi %500 == 0) | ((epi <500) & (epi %50==0))) :
        x = np.linspace(min_list[0],max_list[0],50)
        y = np.linspace(min_list[1],max_list[1],50)
        X,Y = np.meshgrid(x,y)

        if (True):
            Z = np.array([[[Q_for_meshgrid(i,j,k,theta,c) for i in x] for j in y] for k in range(num_action)])
            fig = plt.figure(figsize=(40,10))
            plt.title('episode: %4d,   epsilon: %.3f,   alpha: %.3f,   average_reward: %3d' %(epi, EPSILON, ALPHA, np.mean(reward_list)))

            ax1 = fig.add_subplot(141, projection='3d')
            #plt.gca().invert_zaxis()
            ax1.plot_wireframe(X,Y,Z[0], rstride=1, cstride=1)

            ax2 = fig.add_subplot(142, projection='3d')
            #plt.gca().invert_zaxis()
            ax2.plot_wireframe(X,Y,Z[1], rstride=1, cstride=1)

            ax3 = fig.add_subplot(143, projection='3d')
            #plt.gca().invert_zaxis()
            ax3.plot_wireframe(X,Y,Z[2], rstride=1, cstride=1)


            ax4 = fig.add_subplot(144)
            sns.heatmap(np.argmax(Z, axis=0))
        else:
            fig = plt.figure(figsize=(10,10))

            ax1 = fig.add_subplot(111, projection='3d')
            ax1.set_xlabel('position')
            ax1.set_ylabel('speed')
            ax1.set_zlabel('V')

            ax1.set_title('episode: %4d,   epsilon: %.3f,   alpha: %.3f,   average_reward: %3d' %(epi, EPSILON, ALPHA, np.mean(reward_list)))
            #ax1.set_zlim(-40,0)
            #plt.gca().invert_zaxis()
            Z = np.array([[Q_for_meshgrid(i,j,0,theta,c) for i in x] for j in y])
            ax1.plot_wireframe(X,Y,Z, rstride=2, cstride=2)
            






        plt.savefig('mountaincar_Q_%04d.png' % (epi))
        plt.close()

        reward_list = []



    env.close()

epi=0
while(True):
    done = False
    observation = env.reset() # 環境をリセット
    # 状態を離散化
    s = [int(np.digitize(observation[i], np.linspace(min_list[i],\
            max_list[i], N-1))) for i in range(num_state)]
    # e-greedyによる行動選択   
    a = select_action(Q, s, EPSILON)
    tmp = 0
    while(done == False):
        # 行動aをとり、r, s'を観測
        observation, reward, done, info = env.step(a)
        tmp += reward
        if render:
            env.render()

        # 状態を離散化
        s_dash = [int(np.digitize(observation[i], np.linspace(min_list[i], \
            max_list[i], N-1))) for i in range(num_state)]

        # e-greedyによる行動選択   
        a = select_action(Q, s_dash, EPSILON)
    print("epi: %d, reward: %3d" % (epi, tmp))

plt.plot(reward_array)
plt.show()
