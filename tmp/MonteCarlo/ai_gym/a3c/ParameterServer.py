# -*-coding:utf-8-*-
import tensorflow as tf
import gym, time, random, threading
from gym import wrappers  # gymの画像保存
from keras.models import *
from keras.layers import *
from keras.utils import plot_model
from keras import backend as K
import os

# --グローバルなTensorFlowのDeep Neural Networkのクラスです　-------
class ParameterServer:
    def __init__(self, NUM_STATES, NUM_ACTIONS, LEARNING_RATE, RMSPropDecay):
        self.NUM_STATES = NUM_STATES
        self.NUM_ACTIONS = NUM_ACTIONS
        self.LEARNING_RATE = LEARNING_RATE
        self.RMSPropDecay = RMSPropDecay
        with tf.variable_scope("parameter_server"):      # スレッド名で重み変数に名前を与え、識別します（Name Space）
            self.model = self._build_model()            # ニューラルネットワークの形を決定

        # serverのパラメータを宣言
        self.weights_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="parameter_server")
        self.optimizer = tf.train.RMSPropOptimizer(self.LEARNING_RATE, self.RMSPropDecay)    # loss関数を最小化していくoptimizerの定義です

    # 関数名がアンダースコア2つから始まるものは「外部から参照されない関数」、「1つは基本的に参照しない関数」という意味
    def _build_model(self):     # Kerasでネットワークの形を定義します
        l_input = Input(batch_shape=(None, self.NUM_STATES))
        l_dense = Dense(16, activation='relu')(l_input)
        out_actions = Dense(self.NUM_ACTIONS, activation='softmax')(l_dense)
        out_value = Dense(1, activation='linear')(l_dense)
        model = Model(inputs=[l_input], outputs=[out_actions, out_value])
        #plot_model(model, to_file='A3C.png', show_shapes=True)  # Qネットワークの可視化
        return model


