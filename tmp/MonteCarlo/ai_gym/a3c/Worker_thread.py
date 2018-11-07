# -*-coding:utf-8-*-
import tensorflow as tf
import gym, time, random, threading
from gym import wrappers  # gymの画像保存
from keras.models import *
from keras.layers import *
from keras.utils import plot_model
from keras import backend as K
import os
import Environment

# --スレッドになるクラスです　-------
class Worker_thread:
    # スレッドは学習環境environmentを持ちます
    def __init__(self, thread_name, thread_type, parameter_server, feed_dict):
        self.feed_dict = feed_dict
        self.environment = Environment.Environment(thread_name, thread_type, parameter_server, feed_dict)
        self.thread_type = thread_type

    def run(self):
        while True:
            if not(self.feed_dict['isLearned']) and self.thread_type is 'learning':     # learning threadが走る
                self.environment.run()

            if not(self.feed_dict['isLearned']) and self.thread_type is 'test':    # test threadを止めておく
                time.sleep(1.0)

            if self.feed_dict['isLearned'] and self.thread_type is 'learning':     # learning threadを止めておく
                time.sleep(3.0)

            if self.feed_dict['isLearned']and self.thread_type is 'test':     # test threadが走る
                time.sleep(3.0)
                self.environment.run()




