# coding:utf-8
# -----------------------------------
# OpenGym CartPole-v0 with A3C on CPU
# -----------------------------------
#
# A3C implementation with TensorFlow multi threads.
#
# Made as part of Qiita article, available at
# https://??/
#
# author: Sugulu, 2017

import tensorflow as tf
import gym, time, random, threading
from gym import wrappers  # gymの画像保存
from keras.models import *
from keras.layers import *
from keras.utils import plot_model
from keras import backend as K
import os
import ParameterServer
import Worker_thread

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # TensorFlow高速化用のワーニングを表示させない

# -- constants of Game
ENV = 'CartPole-v0'
ENV = 'MountainCar-v0'
env = gym.make(ENV)
NUM_STATES = env.observation_space.shape[0]     # CartPoleは4状態
NUM_ACTIONS = env.action_space.n        # CartPoleは、右に左に押す2アクション
NONE_STATE = np.zeros(NUM_STATES)

# -- constants of LocalBrain
MIN_BATCH = 5
LOSS_V = .5  # v loss coefficient
LOSS_ENTROPY = .01  # entropy coefficient
LEARNING_RATE = 1e-3
RMSPropDecay = 0.99

# -- params of Advantage-ベルマン方程式
GAMMA = 0.99
N_STEP_RETURN = 5
GAMMA_N = GAMMA ** N_STEP_RETURN

N_WORKERS = 2   # スレッドの数
Tmax = 10   # 各スレッドの更新ステップ間隔

# ε-greedyのパラメータ
EPS_START = 0.5
EPS_END = 0.0
EPS_STEPS = 200*N_WORKERS
TOTAL_STEPS = 300 * N_WORKERS

# -- main ここからメイン関数です------------------------------
# M0.global変数の定義と、セッションの開始です
frames = 0              # 全スレッドで共有して使用する総ステップ数
isLearned = False       # 学習が終了したことを示すフラグ
SESS = tf.Session()     # TensorFlowのセッション開始

feed_dict={'NUM_STATES':NUM_STATES,'NUM_ACTIONS':NUM_ACTIONS,'SESS':SESS,\
        'ENV':ENV, 'LOSS_V':LOSS_V, 'LOSS_ENTROPY':LOSS_ENTROPY,\
        'isLearned':isLearned,'EPS_STEPS':EPS_STEPS,'EPS_START':EPS_START,\
        'EPS_END':EPS_END,'GAMMA':GAMMA,'GAMMA_N':GAMMA_N,\
        'Tmax':Tmax,'N_STEP_RETURN':N_STEP_RETURN,'MIN_BATCH':MIN_BATCH,\
        'NONE_STATE':NONE_STATE, 'TOTAL_STEPS':TOTAL_STEPS}



# M1.スレッドを作成します
with tf.device("/cpu:0"):
    parameter_server = ParameterServer.ParameterServer(NUM_STATES,NUM_ACTIONS, LEARNING_RATE, RMSPropDecay)    # 全スレッドで共有するパラメータを持つエンティティです
    threads = []     # 並列して走るスレッド
    # 学習するスレッドを用意
    for i in range(N_WORKERS):
        thread_name = "local_thread"+str(i+1)
        threads.append(Worker_thread.Worker_thread(thread_name=thread_name, thread_type="learning", parameter_server=parameter_server,feed_dict=feed_dict))

    # 学習後にテストで走るスレッドを用意
    threads.append(Worker_thread.Worker_thread(thread_name="test_thread", thread_type="test", parameter_server=parameter_server, feed_dict=feed_dict))

# M2.TensorFlowでマルチスレッドを実行します
COORD = tf.train.Coordinator()                  # TensorFlowでマルチスレッドにするための準備です
SESS.run(tf.global_variables_initializer())     # TensorFlowを使う場合、最初に変数初期化をして、実行します

running_threads = []
for worker in threads:
    job = lambda: worker.run()      # この辺は、マルチスレッドを走らせる作法だと思って良い
    t = threading.Thread(target=job)
    t.start()
    #running_threads.append(t)

# M3.スレッドの終了を合わせます
#COORD.join(running_threads)
