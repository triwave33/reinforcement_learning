import gym
import numpy as np

env = gym.make('CartPole-v0')
env.reset()

N = 3000

observation_list = []
observation_array = np.zeros((N,4))
for i in range(N):
    if i%100 == 0:
        print("N: %d" % i)
    done = False
    env.reset()
    while(done ==False):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        observation_list.append(observation)
    env.close()

s = np.array(observation_list)

print("minimun")
print(np.min(s, axis=0))
print("maximun")
print(np.max(s, axis=0))

def assign(minimum, maximum, N, value):
    # 区間をN分割するためにはN+1点必要
    linspace = np.linspace(minimum, maximum, N+1)
    for i in range(N):
        if i ==0:
            low = -np.inf
        else:
            low = linspace[i]
        if i == N-1:
            high = np.inf
        else:
            high = linspace[i+1]
        if ((value >= low) & (value <= high)): # high側の等号はinf対策
            ret = i
    return ret

