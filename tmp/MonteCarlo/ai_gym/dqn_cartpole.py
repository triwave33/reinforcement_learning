# coding:utf-8
import gym   
import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import plot_model
from collections import deque
from gym import wrappers  # gymの画像保存
from keras import backend as K
import tensorflow as tf
import QNetwork


# [5] メイン関数開始----------------------------------------------------
# [5.1] 初期設定--------------------------------------------------------
DQN_MODE = 1    # 1がDQN、0がDDQNです
LENDER_MODE = 1 # 0は学習後も描画なし、1は学習終了後に描画する

game = 'CartPole-v0'
env = gym.make(game)
num_episodes = 299  
max_number_of_steps = 200  
goal_average_reward = 195  
num_consecutive_iterations = 10  
total_reward_vec = np.zeros(num_consecutive_iterations)  
GAMMA = 0.99    
EPSILON_INI = 0.9
TAU = 5
islearned = 0  
isrender = 0  
# ---
hidden_size = 16               
learning_rate = 1.E-3
memory_size = 10000            
batch_size = 32                

num_state = 4
num_action = 2

# network
mainQN = QNetwork.QNetwork(hidden_size=hidden_size, learning_rate=learning_rate, state_size=num_state, action_size= num_action)    
targetQN = QNetwork.QNetwork(hidden_size=hidden_size, learning_rate=learning_rate, state_size=num_state, action_size=num_action)  
# plot_model(mainQN.model, to_file='Qnetwork.png', show_shapes=True)        # Qネットワークの可視化
memory = QNetwork.Memory(max_size=memory_size)
actor = QNetwork.Actor()

for episode in range(num_episodes):  
    env.reset()  
    state, reward, done, _ = env.step(env.action_space.sample())  
    state = np.reshape(state, [1, num_state])   
    episode_reward = 0

    
    if episode % TAU == 0:
        targetQN.model.set_weights(mainQN.model.get_weights())

    actions = np.zeros(num_action)
    for t in range(max_number_of_steps + 1):

        #epsilon = 1. - EPSILON_INI * float(episode) / num_episodes
        epsilon = 0.1 + 0.9 / (episode +1.)

        if (islearned == 1) and LENDER_MODE:
            env.render()
            time.sleep(0.1)
            print(state[0, 0])  

        action = actor.get_action(state, epsilon, episode, mainQN)   
        next_state, reward, done, info = env.step(action)   
        next_state = np.reshape(next_state, [1, num_state])  

        actions[action] += 1    # preserve action record

        # 報酬を設定し、与える
        if done:
            next_state = np.zeros(state.shape)  # 次の状態s_{t+1}はない
            if t < 199:
                reward = -1  # 報酬クリッピング、報酬は1, 0, -1に固定
            else:
                reward = 1  # 立ったまま195step超えて終了時は報酬
        else:
            reward = 0  # 各ステップで立ってたら報酬追加（はじめからrewardに1が入っているが、明示的に表す）

        episode_reward += 1   # 合計報酬を更新

        memory.add((state, action, reward, next_state))     # メモリの更新する
        state = next_state  # 状態更新


        # Qネットワークの重みを学習・更新する replay
        if (memory.len() > batch_size) and not islearned:
            mainQN.replay(memory, batch_size, GAMMA, targetQN)

        if DQN_MODE:
            targetQN.model.set_weights(mainQN.model.get_weights())

        # 1施行終了時の処理
        if done:
            total_reward_vec = np.hstack((total_reward_vec[1:], episode_reward))  # 報酬を記録
            print('epi: %d, eps: %f,  t: %d, mean %f, L: %d, R: %d' % \
                (episode, epsilon, t + 1, total_reward_vec.mean(),\
                actions[0], actions[1]))
            break
    
    # terminate condition
    if total_reward_vec.mean() >= goal_average_reward:
        print('Episode %d train agent successfuly!' % episode)
        islearned = 1
        if isrender == 0:   # 学習済みフラグを更新
            isrender = 1

            # env = wrappers.Monitor(env, './movie/cartpoleDDQN')  # 動画保存する場合
            # 10エピソードだけでどんな挙動になるのか見たかったら、以下のコメントを外す
            # if episode>10:
            #    if isrender == 0:
            #        env = wrappers.Monitor(env, './movie/cartpole-experiment-1') #動画保存する場合
            #        isrender = 1
            #    islearned=1;
