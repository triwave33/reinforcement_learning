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
env = gym.make('MountainCar-v0')

num_action_original = env.action_space.n

N =5 # N分割

loads= np.load('theta/constants.npz')
c = loads['c']
b = loads['b']
sigma = loads['sigma']
num_action = loads['num_action']
norm_factor = loads['norm_factor']
theta_path = glob.glob('theta/*theta*.npy')[-1]
print('theta_path: %s' % theta_path)

if num_action_original > num_action:
    factor = 2
else:
    factor = 1

theta = np.load(theta_path)

print('test')

count =0
while(True):
    done = False
    s = env.reset() # 環境をリセット
    # e-greedyによる行動選択   
    a = func.select_action(s,theta,c, 0, num_action, sigma,norm_factor)
    tmp = 0
    while(done == False):
        # 行動aをとり、r, s'を観測
        s, reward, done, info = env.step(a*factor)
        tmp += reward
        env.render()


        # e-greedyによる行動選択   
        a = func.select_action(s,theta,c,0, num_action, sigma, norm_factor)

    print("count: %d, reward: %3d" % (count, tmp))
    count +=1

env.close()
