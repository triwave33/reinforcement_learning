# -*-coding:utf-8-*-
import tensorflow as tf
import gym, time, random, threading
from gym import wrappers  # gymの画像保存
from keras.models import *
from keras.layers import *
from keras.utils import plot_model
from keras import backend as K
import os
import Agent

class Environment:
    total_reward_vec = np.ones(10)*200  # 総報酬を10試行分格納して、平均総報酬をもとめる
    count_trial_each_thread = 0     # 各環境の試行数

    def __init__(self, name, thread_type, parameter_server, feed_dict):
        self.feed_dict = feed_dict
        self.frames= 0
        self.name = name
        self.thread_type = thread_type
        self.env = gym.make(self.feed_dict['ENV'])
        self.agent = Agent.Agent(name, parameter_server, feed_dict)    # 環境内で行動するagentを生成

    def run(self):

        self.agent.brain.pull_parameter_server()  # ParameterSeverの重みを自身のLocalBrainにコピー

        if (self.thread_type is 'test') and (self.count_trial_each_thread == 0):
            self.env.reset()
            #self.env = gym.wrappers.Monitor(self.env, './movie/A3C')  # 動画保存する場合

        s = self.env.reset()
        R = 0
        step = 0
        while True:
            if self.thread_type is 'test':
                self.env.render()   # 学習後のテストでは描画する
                time.sleep(0.01)

            a = self.agent.act(s)   # 行動を決定
            s_, r, done, info = self.env.step(a)   # 行動を実施
            step += 1
            self.frames += 1     # セッショントータルの行動回数をひとつ増やします

            r = 0

            # Cartpole
            if self.feed_dict['ENV'] == 'CartPole-v0':
                if done:  # terminal state
                    self.agent.count += 1
                    s_ = None
                    if step > 198:
                        r = +1
                        print('SUCCEED')
                    else:
                        r = -1

            # MountainCar
            if self.feed_dict['ENV'] == 'MountainCar-v0':
                if done:  # terminal state
                    self.agent.count += 1
                    s_ = None
                    if step < 199:
                        r = +1
                        print('SUCCEED')
                    else:
                        r = -1


            # Advantageを考慮した報酬と経験を、localBrainにプッシュ
            self.agent.advantage_push_local_brain(s, a, r, s_)

            s = s_
            R += r
            if done or (step % self.feed_dict['Tmax'] == 0):  # 終了時がTmaxごとに、parameterServerの重みを更新し、それをコピーする
                if not(self.feed_dict['isLearned']) and self.thread_type is 'learning':
                    self.agent.brain.update_parameter_server()
                    self.agent.brain.pull_parameter_server()

            if done:
                self.total_reward_vec = np.hstack((self.total_reward_vec[1:], step))  # トータル報酬の古いのを破棄して最新10個を保持
                self.count_trial_each_thread += 1  # このスレッドの総試行回数を増やす
                break
        # 総試行数、スレッド名、今回の報酬を出力
        print("#: %d, eps: %.2f, thread:%s, epi:%d, r:%d, mean_r:%.1f" %(self.agent.count, self.agent.eps, self.name ,self.count_trial_each_thread, step, self.total_reward_vec.mean()))

        #  終了判定
        if self.agent.count > self.feed_dict['TOTAL_STEPS']:
            self.feed_dict['isLearned'] = True
            time.sleep(2.0)     # この間に他のlearningスレッドが止まります
            self.agent.brain.push_parameter_server()    # この成功したスレッドのパラメータをparameter-serverに渡します



