### -*-coding:utf-8-*-
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import criff
import pandas as pd

ALPHA = 0.01
ALPHA_INI = 0.1 #decay用の初期値
ALPHA_MIN = 0.01 #decay用の最終値
GAMMA = 1.
EPSILON = 0.1
EPS_INI = 0.1 #decay用の初期値
EPS_MIN = 0.0 #decay用の最終値
num_action = 4
num_episode = 1000 # サンプリング数
ex_factor = 5 # decay完了後も学習を続けるパラメータ
num_row = 4
num_col = 12
ACTIONS = ['right', 'up', 'left', 'down']
DECAY_ALPHA = False
DECAY_EPSILON = False



Q_start_pos = np.zeros((len(ACTIONS),int(num_episode * ex_factor)))


result = np.zeros((2, int(num_episode*ex_factor))) # 0 for sarsa, 1 for Q

def select_action(Q, s, eps):
    # e-greedyによる行動選択   
    greedy = np.argmax(Q[:,s[0],s[1]])
    is_greedy_index = np.where(Q[:,s[0],s[1]] == greedy)[0]
    if len(is_greedy_index) > 1:
        greedy = np.random.choice(is_greedy_index)
    p = [(1- eps + eps/len(ACTIONS)) if i == greedy \
        else eps/len(ACTIONS) for i in range(len(ACTIONS))]

    return  np.random.choice(np.arange(len(ACTIONS)), p=p)
  
##################################################
#sarsa
##################################################

# Qの初期化
Q = np.zeros((num_action, num_row, num_col))
#for epi in tqdm(range(int(num_episode*ex_factor))):
for epi in range(int(num_episode*ex_factor)):
    # greedy方策を徐々に確定的にしていく
    if DECAY_EPSILON:
        EPSILON = max(EPS_MIN, EPS_INI* (1- epi*1./num_episode))
    if DECAY_ALPHA:
        ALPHA = max(ALPHA_MIN, ALPHA_INI* (1- epi*1./num_episode))
    
    # sの初期化
    done = False
    agent = criff.Agent()
    s = agent.get_pos() # 環境をリセット
    # e-greedyによる行動選択   
    a = select_action(Q, s, EPSILON)
    #print("first pos: %d,  %d,  a_dash: %s" % (s[0], s[1],ACTIONS[a]))
    
    tmp = 0 # 報酬積算用
    count = 0

    criff_count = 0
    # エピソードを終端までプレイ
    while(done==False):

        # 行動aをとり、r, s'を観測
        s_dash, reward, done = agent.move(ACTIONS[a])
        #print("s: %d, %d,  a_dash: %s, r: %d, s_dash: %d, %d" % (s[0], s[1],ACTIONS[a],reward, s_dash[0], s_dash[1]))
        if reward == -100:
            criff_count+=1

        tmp += reward

        # e-greedyによる行動選択   
        a_dash = select_action(Q, s_dash, EPSILON)

        # Qの更新
        Q_dash = Q[a_dash, s_dash[0], s_dash[1]]
        Q[a,s[0],s[1]] += ALPHA*(reward + GAMMA*(Q_dash)\
                                    - Q[a,s[0],s[1]])
        s = s_dash
        a = a_dash

        count += 1

        if count>200:
            done = True

    print("epi: %d, eps#: %.3f, alpha: %.3f,  reward: %3d, criff_count: %d" % (epi, EPSILON, ALPHA, tmp, criff_count))
    result[0, epi] = tmp

    Q_start_pos[:,epi] = Q[:,3,0]
    Q_for_sarsa = Q

##################################################
#Q-learning
##################################################

# Qの初期化
Q = np.zeros((num_action, num_row, num_col))
#for epi in tqdm(range(int(num_episode*ex_factor))):
for epi in range(int(num_episode*ex_factor)):
    # greedy方策を徐々に確定的にしていく
    if DECAY_EPSILON:
        EPSILON = max(EPS_MIN, EPS_INI* (1- epi*1./num_episode))
    if DECAY_ALPHA:
        ALPHA = max(ALPHA_MIN, ALPHA_INI* (1- epi*1./num_episode))
    
    # sの初期化
    done = False
    agent = criff.Agent()
    s = agent.get_pos() # 環境をリセット

    # e-greedyによる行動選択   
    a = select_action(Q, s, EPSILON)
    #print("first pos: %d,  %d,  a_dash: %s" % (s[0], s[1],ACTIONS[a]))
    
    tmp = 0 # 報酬積算用
    count = 0

    criff_count = 0
    # エピソードを終端までプレイ
    while(done==False):
        # e-greedyによる行動選択   
        a = select_action(Q, s, EPSILON)
        #print("first pos: %d,  %d,  a_dash: %s" % (s[0], s[1],ACTIONS[a]))
 
        # 行動aをとり、r, s'を観測
        s_dash, reward, done = agent.move(ACTIONS[a])
        #print("s: %d, %d,  a_dash: %s, r: %d, s_dash: %d, %d" % (s[0], s[1],ACTIONS[a],reward, s_dash[0], s_dash[1]))
        if reward == -100:
            criff_count+=1

        tmp += reward

        # Qの最大値をとる行動を選択   
        a_dash = select_action(Q, s_dash, 0)

        # Qの更新
        Q_dash = Q[a_dash, s_dash[0], s_dash[1]]
        Q[a,s[0],s[1]] += ALPHA*(reward + GAMMA*(Q_dash)\
                                    - Q[a,s[0],s[1]])
        s = s_dash

        count += 1

        if count>200:
            done = True

    print("epi: %d, eps#: %.3f, alpha: %.3f,  reward: %3d, criff_count: %d" % (epi, EPSILON, ALPHA, tmp, criff_count))
    result[1, epi] = tmp

    Q_start_pos[:,epi] = Q[:,3,0]
    Q_for_q = Q


fig = plt.figure(figsize=(13,8))
plt.plot(result[0,:], label='sarsa', alpha=.3, c='b')
N = 50
plt.plot(pd.Series(result[0,:]).rolling(window=N).mean(), label='sarsa', c='b')
plt.plot(result[1,:], label='Q', alpha=.3, c='r')
plt.plot(pd.Series(result[1,:]).rolling(window=N).mean(), label='Q', c='r')
plt.legend(loc='best')
plt.show()
criff.Q_arrow_plot(Q_for_sarsa)
plt.title('Q_for_sarsa')
plt.show()
criff.Q_arrow_plot(Q_for_q)
plt.title('Q_for_q')
plt.show()


#plt.subplot(2,2,3)
#fig = plt.figure(figsize=(13,8))
#plt.show()
#criff.Q_value_plot(Q)

#for i in range(len(ACTIONS)):
#    plt.subplot(4,1,i+1)
#    plt.plot(Q_start_pos[i])
#plt.show()

#while(False):
#    done = False
#    observation = env.reset() # 環境をリセット
#    # 状態を離散化
#    s = [int(np.digitize(observation[i], np.linspace(min_list[i],\
#            max_list[i], N-1))) for i in range(num_state)]
#    # e-greedyによる行動選択   
#    a = select_action(Q, s, EPSILON)
#    while(done == False):
#        # 行動aをとり、r, s'を観測
#        observation, reward, done, info = env.step(a)
#        env.render()
#
#        # 状態を離散化
#        s_dash = [int(np.digitize(observation[i], np.linspace(min_list[i], \
#            max_list[i], N-1))) for i in range(num_state)]
#
#        # e-greedyによる行動選択   
#        a = select_action(Q, s_dash, EPSILON)
#

