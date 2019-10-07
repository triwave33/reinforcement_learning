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
import glob
import func 
import time
import os
import shutil

env = gym.make('MountainCar-v0')

num_action_original = env.action_space.n

N =3 # N分割
play_count = 100

use_ext_disk = False

if use_ext_disk:
    loadpath = '/volumes/data/dataset/ai_gym/'
else:
    loadpath = './'

loads= np.load(loadpath +'theta/constants.npz')
c = loads['c']
b = loads['b']
sigma = loads['sigma']
num_action = loads['num_action']
norm_factor = loads['norm_factor']

def select_theta(num = ''):
    if num=='':
        theta_path = sorted(glob.glob(loadpath + 'theta/*theta*.npy'),\
                    key = os.path.getmtime)
        theta_path = theta_path[-1]
    else:
        theta_path = glob.glob(loadpath + 'theta/*theta*' + num + '.npy')[0]
    return theta_path

theta_path = select_theta('')        

print('theta_path: %s' % theta_path)

if num_action_original > num_action:
    factor = 2
else:
    factor = 1

theta = np.load(theta_path)


while(True):
    count =0
    print('theta_path: %s' % theta_path)
    while(count <play_count):
        done = False
        s = env.reset() # 環境をリセット
        # e-greedyによる行動選択   
        a = func.select_action(s,theta,c, 0, num_action, sigma,norm_factor)
        tmp = 0
        for i in tqdm(range(200)):
            # 行動aをとり、r, s'を観測
            s, reward, done, info = env.step(a*factor)
            tmp += reward
            env.render()
            if done:
                break
    
    
            # e-greedyによる行動選択   
            a = func.select_action(s,theta,c,0, num_action, sigma, norm_factor)
    
        print("count: %d, reward: %3d" % (count, tmp))
        count +=1
    
    print('now waiting')
    while (theta_path == select_theta('')):
        time.sleep(1)
    theta_path = select_theta('')
    theta = np.load(theta_path)


env.close()
