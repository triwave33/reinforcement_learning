### -*-coding:utf-8-*-
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tqdm import tqdm
import  gym
import math
import seaborn as sns

# Ai GymのCartPoleを使用
env = gym.make('MountainCar-v0')

# 状態空間を次の様に分割
N =32 # N分割
min_list = env.observation_space.low  # 下限値 [-1.2 -0.07]
max_list = env.observation_space.high  # 上限値 [-0.6, 0.07]

min_list = np.array([-1.2, -0.07])  # 下限値
max_list = np.array([0.6,0.07])  # 上限値



ACTIONS = [0,1,2]
ALPHA = 0.1
ALPHA_INI = 0.5
ALPHA_LAST = 0.0
GAMMA = 0.99
EPS_INI = 0.0
EPS_LAST = 0.00
num_state = 2
num_action = 3
num_episode = 1601 # サンプリング数
render = 0 # 描写モード
ex_factor = 1.0 # epsilonがゼロになったあとも学習を続けるパラメータ
render_interval = 25

def select_action(Q, s, eps):
    # e-greedyによる行動選択   
    greedy = np.argmax(Q[:,s[0],s[1]])
    p = [(1- eps + eps/len(ACTIONS)) if i == greedy \
        else eps/len(ACTIONS) for i in range(len(ACTIONS))]
    return  np.random.choice(np.arange(len(ACTIONS)), p=p)
 


result = []
for trial in range(5):
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
    
    #for epi in tqdm(range(int(num_episode*ex_factor))):
    reward_list = []
    min_pos = min_list[0]
    max_pos = max_list[1]

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



        action_result = np.zeros(num_action) 
        # エピソードを終端までプレイ
        while(done==False):

            #if count %20 == 0:
            #    x = np.linspace(min_list[0],max_list[0],N)
            #    y = np.linspace(min_list[1],max_list[1],N)
            #    X,Y = np.meshgrid(x,y)

            #    fig = plt.figure(figsize=(10,10))
            #    ax1 = fig.add_subplot(111, projection='3d')
            #    ax1.set_xlabel('x')
            #    #ax1.set_xlim(-0.5,0.5)
            #    #ax1.set_ylim(-0.5,0.5)
            #    ax1.set_zlim(-10,0)
            #    plt.gca().invert_zaxis()
            #    ax1.plot_wireframe(X,Y,Q[0,:,:], rstride=1, cstride=1)
            #    #ax1.plot_wireframe(X,Y,Q[0,:,:], rstride=max(1,2**(up-1)), cstride=max(1,2**(up-1)))
            #    #ax1.plot_surface(X,Y,Q[0,:,:], cmap='bwr', linewidth=0)
            #    #sns.heatmap(Q[0,:,:])
            #    plt.savefig('mountaincar_Q_%d_%04d_%03d.png' % (up, epi, count))
            #    plt.close()


            if render:
                env.render()
    
            # e-greedyによる行動選択   
            a = select_action(Q, s, EPSILON)
            action_result[a] +=1
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

        if ((epi %render_interval == 0) ) :
            x = np.linspace(min_list[0],max_list[0],N)[1:-1]
            y = np.linspace(min_list[1],max_list[1],N)[1:-1]
            X,Y = np.meshgrid(x,y)

            if (False):
                fig = plt.figure(figsize=(40,10))
                plt.title('episode: %4d,   epsilon: %.3f,   alpha: %.3f,   average_reward: %3d' %(epi, EPSILON, ALPHA, np.mean(reward_list)))

                ax1 = fig.add_subplot(141, projection='3d')
                #ax1.set_zlim(-40,0)
                plt.gca().invert_zaxis()
                ax1.plot_wireframe(X,Y,Q[0,1:-1,1:-1], rstride=2, cstride=2)

                ax2 = fig.add_subplot(142, projection='3d')
                #ax2.set_zlim(-40,0)
                plt.gca().invert_zaxis()
                ax2.plot_wireframe(X,Y,Q[1,1:-1,1:-1], rstride=2, cstride=2)

                ax3 = fig.add_subplot(143, projection='3d')
                #ax3.set_zlim(-40,0)
                plt.gca().invert_zaxis()
                ax3.plot_wireframe(X,Y,Q[2,1:-1,1:-1], rstride=2, cstride=2)

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
                plt.gca().invert_zaxis()
                #ax1.plot_wireframe(X,Y,np.mean(Q,axis=0)[1:-1,1:-1], rstride=1, cstride=1)
                ax1.plot_wireframe(X,Y,Q[2,1:-1,1:-1], rstride=2, cstride=2)
                



            plt.savefig('mountaincar_Q_%d_%04d.png' % (trial, epi))
            plt.close()

        #    reward_list = []


 
        print("N:%d, epi:%d, eps#:%.3f, reward:%3d, min_pos:%d, max_pos:%d, [%d,%d,%d]" \
            % (N, epi, EPSILON, tmp, min_pos, max_pos,\
            action_result[0], action_result[1], action_result[2]))
        Q_10trial[trial, epi] = tmp
        reward_array[epi] = tmp
        env.close()
    result.append(reward_list)

np.save('tableQ.npy', result)
N *= 2
Q = Q.repeat(2,axis=1).repeat(2,axis=2)
visit_array = visit_array.repeat(2,axis=1).repeat(2,axis=2)





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
