# -*-coding:utf-8-*-
import tensorflow as tf
import gym, time, random, threading
from gym import wrappers  # gymの画像保存
from keras.models import *
from keras.layers import *
from keras.utils import plot_model
from keras import backend as K
import LocalBrain

# --行動を決定するクラスです、CartPoleであれば、棒付き台車そのものになります　-------
class Agent:
    def __init__(self, name, parameter_server, feed_dict):
        self.eps = 0
        self.feed_dict=feed_dict
        self.brain = LocalBrain.LocalBrain(name, parameter_server, feed_dict)   # 行動を決定するための脳（ニューラルネットワーク）
        self.memory = []        # s,a,r,s_の保存メモリ、　used for n_step return
        self.R = 0.             # 時間割引した、「いまからNステップ分あとまで」の総報酬R
        self.count = 0

    def act(self, s):
        if self.feed_dict['isLearned']:
            self.eps = 0
        elif self.count >= self.feed_dict['EPS_STEPS']:   # ε-greedy法で行動を決定します 171115修正
            self.eps = self.feed_dict['EPS_END']
        else:
            self.eps = self.feed_dict['EPS_START'] + self.count*1. * (self.feed_dict['EPS_END'] - self.feed_dict['EPS_START']) / self.feed_dict['EPS_STEPS']  # linearly interpolate
            #eps = EPS_START +  (EPS_END - EPS_START) / EPS_STEPS   # linearly interpolate

        if random.random() < self.eps:
            return random.randint(0, self.feed_dict['NUM_ACTIONS'] - 1)   # ランダムに行動
        else:
            s = np.array([s])
            p = self.brain.predict_p(s)

            # a = np.argmax(p)  # これだと確率最大の行動を、毎回選択

            a = np.random.choice(self.feed_dict['NUM_ACTIONS'], p=p[0])
            # probability = p のこのコードだと、確率p[0]にしたがって、行動を選択
            # pにはいろいろな情報が入っていますが確率のベクトルは要素0番目
            return a

    def advantage_push_local_brain(self, s, a, r, s_):   # advantageを考慮したs,a,r,s_をbrainに与える
        def get_sample(memory, n):  # advantageを考慮し、メモリからnステップ後の状態とnステップ後までのRを取得する関数
            s, a, _, _ = memory[0]
            _, _, _, s_ = memory[n - 1]
            return s, a, self.R, s_

        # one-hotコーディングにしたa_catsをつくり、、s,a_cats,r,s_を自分のメモリに追加
        a_cats = np.zeros(self.feed_dict['NUM_ACTIONS'])  # turn action into one-hot representation
        a_cats[a] = 1
        self.memory.append((s, a_cats, r, s_))

        # 前ステップの「時間割引Nステップ分の総報酬R」を使用して、現ステップのRを計算
        self.R = (self.R + r * self.feed_dict['GAMMA_N']) / self.feed_dict['GAMMA']     # r0はあとで引き算している、この式はヤロミルさんのサイトを参照

        # advantageを考慮しながら、LocalBrainに経験を入力する
        if s_ is None:
            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, r, s_ = get_sample(self.memory, n)
                self.brain.train_push(s, a, r, s_)
                self.R = (self.R - self.memory[0][2]) / self.feed_dict['GAMMA']
                self.memory.pop(0)

            self.R = 0  # 次の試行に向けて0にしておく

        if len(self.memory) >= self.feed_dict['N_STEP_RETURN']:
            s, a, r, s_ = get_sample(self.memory, self.feed_dict['N_STEP_RETURN'])
            self.brain.train_push(s, a, r, s_)
            self.R = self.R - self.memory[0][2]     # # r0を引き算
            self.memory.pop(0)



