import numpy as np
import time
from collections import Counter
import gym
import matplotlib.pyplot as plt
import seaborn as sns
gym.logger.set_level(40)


env = gym.make('MountainCarContinuous-v0')

INERTIA = 0.0  # 0.99
VELOCITY_LIMIT = 0.5 # 0.1
SIGMA = 0.2 #1.0
X_BONUS = 10.0
ALPHA = 1.5E+1
ITER_NUM = 10000
DECAY_RATE = 0.99
EPSILON_INITIAL = 0.8
use_momentum = False
use_positional_reward = True
num_play = 10
sum_r_list = []
theta_list = []
def policy_random(state):
    return np.random.normal(size=1)


class Policy(object):
    def __init__(self):
        self.theta = np.random.normal(scale=0.1, size=(2, 1)) 
        #self.theta = np.array([-0.5, 0]).reshape(2,1)
        #self.theta = np.random.normal(scale=0.001, size=(2, 1)) 
        self.rmsprop_memory = np.zeros((2,1))
        self.sum_succeed=0

    def __call__(self, state):
        mean = state.dot(self.theta)
        a = np.random.normal(mean, SIGMA)
        #print(epsilon)
        if np.random.rand() < self.epsilon:
            #a = (np.random.rand() -0.5)*2
            a = np.random.randint(-1,2)
            a = [a]
        else:
            if a<-1:
                a=[-1]
            elif a>1:
                a=[1]
        return a

    def get_action(self, state, eps):
        mean = state.dot(self.theta)
        a = np.random.normal(mean, SIGMA)
        #print(epsilon)
        if np.random.rand() < eps:
            #print("epsilon!!!")
            a = (np.random.rand() -0.5)*2
            a = [a]
            #print(a)
        else:
            if a<-1:
                a=[-1]
            elif a>1:
                a=[1]
        #print(a)
        return a

    def grad(self, state, action):
        t1 = -(action - state.dot(self.theta))
        # 2
        t2 = -state
        # 4
        g = np.outer(t2, t1)
        return g


def play(policy, epsilon, num_plays=1, to_print=False):
    print("playing")
    for i in range(num_plays):
        s = env.reset()
        while True:
            env.render()
            a = policy.get_action(s,epsilon)
            s, r, done, info = env.step(a)
            if done: 
                break #print env.result_log[-1]
    #env.close()


def reinforce(policy, epsilon,  num_plays=num_play, to_print=False, render=False):
    #env = gym.make('MountainCar-v0')
    
    result = 0
    samples = []
    sum_t = 0
    sum_r = 0.0
    sum_left = 0
    sum_right = 0
    for i in range(num_plays):
        s = env.reset()
        t = 0
        SARs = []
        while True:
            #env.render()
            #s = env.get_state()
            a = policy.get_action(s,epsilon)
            if a[0]<0:
                sum_left+=1
            elif a[0]>0:
                sum_right+=1
            s, r, done, info = env.step(a)
            
            if use_positional_reward:
                if s[0]>=0:
                    r = abs(s[0])
                else:
                    r = abs(s[0]) *0.5


            if done:
                if t <998:
                    r = 1000000./t
                    print("####SUCCEED##### reward:%f" % r)
                    policy.sum_succeed+=1


            t += 1
            sum_r += r
            SARs.append((s, a, r))
            if done:
                break
        #env.close()
        samples.append((t, SARs))
        sum_t += t
        #print(sum_t)

    baseline = float(sum_r) / sum_t # 
    #baseline = -0.5
    grad = np.zeros((2, 1))
    for (t, SARs) in samples:
        tmp_grad = np.zeros((2, 1))
        for (s, a, r) in SARs:
            g = policy.grad(s, a)
            tmp_grad += g * (r - baseline)
        grad += tmp_grad / t
    grad /= num_plays
    #policy.theta /= np.linalg.norm(policy.theta)
    
    if np.linalg.norm(grad) > 1:
        grad /= np.linalg.norm(grad)
    theta_list.append(policy.theta.copy())
    #policy.theta += ALPHA * grad
    #policy.theta += ALPHA * grad  # +?

    ### RMS prop ###
    policy.rmsprop_memory = DECAY_RATE * policy.rmsprop_memory + (1 - DECAY_RATE) * grad**2
    
    if use_momentum:
        policy.theta  += ALPHA * grad / (np.sqrt(policy.rmsprop_memory) + 1e-5)
    else:
        policy.theta += ALPHA * grad

    #policy.theta  += ALPHA * grad / (np.sqrt(policy.rmsprop_memory) + 1e-5)
    print 'theta'
    print policy.theta
    print 'grad'
    print grad
    print("baseline: %.5f, sum_t: %d" % (baseline, sum_t))
    print("sum_left: %d, sum_right: %d" % (sum_left, sum_right))
    print("sum_r: %.5d" % sum_r)
    print("epsilon: %.5f" % epsilon)
    print("sum_succeed: %d" % policy.sum_succeed)
    sum_r_list.append(sum_r)
    return samples, sum_r_list, theta_list


#print Counter(play(policy_random).result_log)
#print Counter(play(Policy()).result_log)
policy = Policy()
for i in range(ITER_NUM):
    print
    print(i)
    epsilon = float(ITER_NUM -i)/ITER_NUM * EPSILON_INITIAL
    samples, sum_r_list, theta_list = reinforce(policy, epsilon, num_play)

    if i % 50 == 0:
        #play(policy, epsilon, 1)
        low1, low2 = env.observation_space.low
        high1, high2 = env.observation_space.high
        x = np.linspace(low1,high1,50)
        y = np.linspace(low2,high2,100)
        X,Y = np.meshgrid(x,y)
        Z = policy.theta[0]*X + policy.theta[1]*Y
 
        color_pos = np.array(sum_r_list) > 0
        color_neg = np.array(sum_r_list) < 0
        theta_array = np.array(theta_list)
        theta1 = theta_array[:,0,0]
        theta2 = theta_array[:,1,0]
        plt.scatter(theta1[color_pos], theta2[color_pos], c='r', marker='.')
        plt.scatter(theta1[color_neg], theta2[color_neg], c='k', marker='.')
        plt.plot(theta1, theta2, c='k')
        plt.savefig('fig_%05d.png' % i)
        plt.close()
        sns.heatmap(Z, xticklabels='auto', yticklabels='auto',vmin=-1,vmax=1)
        plt.savefig('mesh_%05d.png' % i)
        plt.close()
        plt.plot(sum_r_list)
        plt.savefig('sum_r%05d.png' % i)
        plt.close()

    #env.close()
       
    #print Counter(env.result_log)

    #from PIL import Image, ImageDraw
    #im = Image.new('RGB', (300, 200), color=(255,255,255))
    #d = ImageDraw.Draw(im)
    #for t, SARs in samples:
    #    points = [(START_X, START_Y)]
    #    for s, a, r in SARs:
    #        points.append(tuple(s[:2]))
    #    d.line(points, fill=0)
    #d.rectangle((100, 30, 200, 150), fill=(128, 128, 128))
    #im.save('reinforce{:04d}.png'.format(i))

color_pos = np.array(sum_r_list) > 0
color_neg = np.array(sum_r_list) < 0
theta_array = np.array(theta_list)
theta1 = theta_array[:,0,0]
theta2 = theta_array[:,1,0]
plt.scatter(theta1[color_pos], theta2[color_pos], c='r', marker='.')
plt.scatter(theta1[color_neg], theta2[color_neg], c='k', marker='.')
plt.plot(theta1, theta2, c='k')
plt.show()

plt.scatter(range(len(theta1)), theta1,  marker='.', c=color_pos)
plt.show()

plt.scatter(range(len(sum_r_list)), sum_r_list,  marker='.', c=color_pos)
plt.show()


