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
ALPHA_INI = 1.0E-2
ALPHA_LAST = 0
GAMMA = 0.9
EPS_INI = 0.1
EPS_LAST = 0
LAMBDA = 0.6
render = 0 # 描写モード
ex_factor = 1.5 # epsilonがゼロになったあとも学習を続けるパラメータ
sigma=0.3
use_potential_reward = False # 位置に応じた報酬
use_binary_action = False # 左・右のみのアクション
num_episode = 3000

N =4 # N分割


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
    #c[2+i,(i*num_action**2):(i+1)*num_action**2] =1
    c[2+i,(i*b):((i+1)*b)] =1
b *= num_action


print_log = False

# 基底関数
norm_factor = np.append(np.array([1.2, 0.07]), np.ones(num_action))

def rbf(s,a, cb, sigma, num_action):
    a = one_hot(a, num_action)
    x = np.hstack([s,a])
    return np.exp(-np.square(LA.norm((x-cb)/norm_factor))/(2*sigma**2)) 

np.savez('theta/constants',c=c, b=b, sigma=sigma, num_action=num_action, norm_factor=norm_factor)
# 初期化
theta = np.random.rand(b) # 初期化
theta = np.zeros(b) # 初期化
e = np.zeros(b)

# Q
def Q(s, a, theta,c,sigma,num_action):
    res =0
    for i in range(c.shape[1]):
        cb = c[:,i]
        res += theta[i] * rbf(s,a,cb,sigma,num_action)
    return res

# Q
def Q_for_meshgrid(s0,s1, a, theta,c,sigma):
    s = np.hstack([s0,s1])
    rbfs = np.array([rbf(s,a,c[:,i],sigma,num_action) for i in range(c.shape[1])])
    return theta.dot(rbfs)


def one_hot(a, num_action):
    assert a <num_action
    array = np.zeros(num_action)
    array[a] = 1
    return array

def select_action(s,theta,c, eps, num_action,sigma):
    # e-greedyによる行動選択   
    if np.random.rand() > eps:
        qs = [Q(s, i, theta, c, sigma, num_action)\
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

    # initialize s
    s = env.reset() # 環境をリセット
    
    # e-greedyによる行動選択   
    a = select_action(s, theta, c, EPSILON, num_action, sigma)

    tmp = 0 # 報酬積算用
    
    x = np.array([rbf(s,a,c[:,i],sigma, num_action) for i in range(c.shape[1])])

    count = 0
    Q_val_old = 0
    z = np.zeros(b)

    while(done==False):
        if render:
            env.render()
        

        # 行動aをとり、r, s'を観測
        if use_binary_action:
            s_dash, reward, done, info = env.step(a*2)
        else:
            s_dash, reward, done, info = env.step(a)


        if use_potential_reward:
            reward = s_dash[0]**2
            if s[0]>0:
                reward*=2


        a_dash = select_action(s_dash, theta, c, EPSILON,num_action, sigma)
        
        x_dash = np.array([rbf(s_dash,a_dash,c[:,i],sigma,num_action) \
                for i in range(c.shape[1])])

        Q_val = theta.dot(x)
        Q_val_dash = theta.dot(x_dash)

        DELTA = reward + GAMMA*Q_val_dash - Q_val

        z = GAMMA * LAMBDA * z + (1- ALPHA * GAMMA * LAMBDA * ((z.T).dot(x)))*x

        theta = theta + ALPHA*(DELTA + Q_val - Q_val_old)*z - ALPHA*(Q_val - Q_val_old)*x

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
    memory = np.array(visit_list)
    
    print( theta)
    print("epi: %d, eps: %f, alpha: %f, TDerr: %f  reward %f: " % (epi, EPSILON, ALPHA, DELTA, np.mean(reward_list)))
    print(action_result)

            
            
            

    if ((epi %500 == 0) | ((epi <2000) & (epi %50==0))) :
        x = np.linspace(min_list[0],max_list[0],50)
        y = np.linspace(min_list[1],max_list[1],50)
        X,Y = np.meshgrid(x,y)

        if (True):
            Z = np.array([[[Q_for_meshgrid(i,j,k,theta,c,sigma) for i in x] for j in y] for k in range(num_action)])
            fig = plt.figure(figsize=(40,10))
            plt.title('episode: %4d,   epsilon: %.3f,   alpha: %.3f,   average_reward: %3d' %(epi, EPSILON, ALPHA, np.mean(reward_list)))

            ax1 = fig.add_subplot(141, projection='3d')
            #plt.gca().invert_zaxis()
            ax1.plot_wireframe(X,Y,Z[0], rstride=1, cstride=1)

            ax2 = fig.add_subplot(142, projection='3d')
            #plt.gca().invert_zaxis()
            ax2.plot_wireframe(X,Y,Z[1], rstride=1, cstride=1)
            
            if use_binary_action != True:
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
            






        np.save('theta/mountaincar_Q_theta_%04d.npy' % (epi), theta)
        plt.savefig('fig/mountaincar_Q_%04d.png' % (epi))
        plt.close()


    env.close()
plt.plot(reward_array)
plt.show()
