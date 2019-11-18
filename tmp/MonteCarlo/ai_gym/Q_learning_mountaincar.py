### -*-coding:utf-8-*-
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tqdm import tqdm
import  gym
import math
import seaborn as sns

env_name = 'MountainCar-v0'
env_name = 'CartPole-v0'

# Ai GymのCartPoleを使用
env = gym.make(env_name)
print("env: %s" % env_name)

if env_name == 'MountainCar-v0':
    min_list = np.array([-1.2, -0.04])  # 下限値
    max_list = np.array([0.6,0.07])  # 上限値
    num_state = 2
    num_action = 3
    plot = True # 図への描画
elif env_name == 'CartPole-v0':
    min_list = np.array([-4.8, -3.4, -0.4, -3.4])  # 下限値
    max_list = np.array([4.8, 3.4, 0.4, 3.4])  # 上限値
    num_state = 4
    num_action = 2
    plot = False # plotはMountainCarの時のみ有効
       
# 状態空間を次の様に分割
N = 32

ALPHA = 0.1
ALPHA_INI = 0.1
ALPHA_LAST = 0.1
GAMMA = 0.99
EPS_INI = 0.5
EPS_LAST = 0.0
num_episode = 50000 # サンプリング数

### mode ###
learning = True
render = False # 描写モード
ex_factor = 1.0 # epsilonがゼロになったあとも学習を続けるパラメータ


def select_action(Q, s, eps):
    # e-greedyによる行動選択   
    greedy = np.argmax(Q[(...,  *tuple(s))])
    p = [(1- eps + eps/num_action) if i == greedy \
        else eps/num_action for i in range(num_action)]
    return  np.random.choice(np.arange(num_action), p=p)
   

def main():
    
    # 分割数 
    N = 32

    # Qの初期化
    Q = np.zeros(((num_action,) + (N,)*num_state)) 
    Q_10trial = np.zeros((10, int(num_episode*ex_factor)))
    
    # 訪問回数を格納する配列
    visit_array = np.zeros(((num_action,) + (N,)*num_state))
    # 収益を格納する配列
    reward_array = np.zeros(int(num_episode * ex_factor))
    
    # epsilon-greedyのための確率分布を初期化
    p = np.random.rand(num_action)
    p = p / np.sum(p, axis=0) # 正規化


    for trial in range(1):
        #for epi in tqdm(range(int(num_episode*ex_factor))):
        reward_list = []
        min_pos = min_list[0]
        max_pos = max_list[0]
        

        for epi in range(int(num_episode*ex_factor)):
            # greedy方策を徐々に確定的にしていく
            EPSILON = max(EPS_LAST, EPS_INI* (1- epi*1./num_episode))
            ALPHA = max(ALPHA_LAST, ALPHA_INI* (1- epi*1./num_episode))
            
            # sの初期化
            done = False
            observation = env.reset() # 環境をリセット
            # 状態を離散化
            s = [int(np.digitize(observation[i], np.linspace(min_list[i],\
                    max_list[i], N-1))) for i in range(num_state)]
            
            tmp = 0 # 報酬積算用
            count = 0

 
            # エピソードを終端までプレイ
            while(done==False):
 
                if render:
                    env.render()
        
                # e-greedyによる行動選択   
                a = select_action(Q, s, EPSILON)
                visit_array[a, s[0], s[1]] +=1
        
                # 行動aをとり、r, s'を観測
                observation, reward, done, info = env.step(a)
        
                # 状態を離散化
                s_dash = [int(np.digitize(observation[i], np.linspace(min_list[i], \
                    max_list[i], N-1))) for i in range(num_state)]

                if min_pos > observation[0]:
                    min_pos = observation[0]
                if max_pos < observation[0]:
                    max_pos = observation[0]
                
                tmp += reward
        
                # argmaxによる行動選択(Q_learning)   
                Q_dash = np.max(Q[:,s_dash[0],s_dash[1]])
        
                # Qの更新
                Q[tuple([a]) + tuple(s)] += ALPHA*(reward + GAMMA*(Q_dash)\
                                            - Q[tuple([a]) + tuple(s)])
                s = s_dash
        
                count += 1

            reward_list.append(tmp)

            if epi %500 == 0:
                if plot:
                    make_plot(epi, EPSILON, reward_list, Q)
     
            print("N: %d, epi: %d, eps#: %.3f, reward: %3d" % (N, epi, EPSILON, tmp))
            Q_10trial[trial, epi] = tmp
            reward_array[epi] = tmp
            env.close()
    N *= 2
    Q = Q.repeat(2,axis=1).repeat(2,axis=2)
    visit_array = visit_array.repeat(2,axis=1).repeat(2,axis=2)

    reward_list = []


def make_plot(epi, EPSILON, reward_list, Q):
    x = np.linspace(min_list[0],max_list[0],N)[1:-1]
    y = np.linspace(min_list[1],max_list[1],N)[1:-1]
    X,Y = np.meshgrid(x,y)

    if (False):
        fig = plt.figure(figsize=(40,10))
        plt.title('episode: %4d,   epsilon: %.3f,   alpha: %.3f,   average_reward: %3d' %(epi, EPSILON, ALPHA, np.mean(reward_list)))

        ax1 = fig.add_subplot(141, projection='3d')
        ax1.set_zlim(-40,0)
        plt.gca().invert_zaxis()
        ax1.plot_wireframe(X,Y,Q[0,:,:], rstride=1, cstride=1)

        ax2 = fig.add_subplot(142, projection='3d')
        ax2.set_zlim(-40,0)
        plt.gca().invert_zaxis()
        ax2.plot_wireframe(X,Y,Q[1,:,:], rstride=2, cstride=2, linewidth=.1)

        ax3 = fig.add_subplot(143, projection='3d')
        ax3.set_zlim(-40,0)
        plt.gca().invert_zaxis()
        ax3.plot_wireframe(X,Y,Q[2,:,:], rstride=2, cstride=2)

        ax4 = fig.add_subplot(144)
        sns.heatmap(np.argmax(Q, axis=0))
    else:
        fig = plt.figure(figsize=(10,10))

        ax1 = fig.add_subplot(111, projection='3d')
        ax1.set_xlabel('position')
        ax1.set_ylabel('speed')
        ax1.set_zlabel('V')

        ax1.set_title('episode: %4d,   epsilon: %.3f,   alpha: %.3f,   average_reward: %3d' %(epi, EPSILON, ALPHA, np.mean(reward_list)))
        #ax1.set_zlim(-40,0)
        #plt.gca().invert_zaxis()
        ax1.plot_wireframe(X,Y,np.mean(Q,axis=0)[1:-1,1:-1], rstride=1, cstride=1)

    plt.savefig('images/mountaincar_Q_%04d.png' % (epi))
    plt.close()



if __name__ == "__main__":
    main()
