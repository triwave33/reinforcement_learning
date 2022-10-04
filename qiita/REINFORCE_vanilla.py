
import gym
import numpy as np
from torch import nn
import torch
from torch.distributions import Categorical
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

env = gym.make('CartPole-v0')
num_episode = 1000
eps_ini = 0
eps_last = 0
alpha_ini = 0.001
alpha_last= 1e-4

render= False
eps = np.finfo(np.float32).eps.item()

FEATURE_DIM = 4
NUM_ACTION = 2
GAMMA = 0.99
#ALPHA = 1e-4

def get_G_list(r_list, normalize=False): # get
    R = 0
    G_list = []
    for r in r_list[::-1]:
        R = r + GAMMA*R
        G_list.insert(0, R)
    G_list = np.array(G_list)
    if normalize:
        (G_list - G_list.mean()) / (G_list.std() + eps)
    return G_list
    

def get_feature(s):
    return torch.tensor([s**i for i in range(5)]).reshape(-1,1)


class Softmax_policy_net(nn.Module):
    def __init__(self):
        super(Softmax_policy_net, self).__init__()
        self.num_action = NUM_ACTION
        self.feature_dimension = FEATURE_DIM
        self.linear1 = nn.Linear(in_features= self.feature_dimension, 
                                out_features= 128,
                                bias = True)
        self.linear2 = nn.Linear(128, NUM_ACTION)
        self.dropout = nn.Dropout(p=0.6)
        self.log_probs_list = []


    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.linear2(x)
        return F.softmax(x, dim=1)



    def update(self, G, t):
        for param in self.parameters():
            param.data += ALPHA * (GAMMA**t) * G * param.grad
            param.grad.zero_()

    def update_all(self):
        for param in self.parameters():
            param.data += ALPHA * param.grad
            param.grad.zero_()

    def finish_episode(self, r_list):

        R = 0
        policy_loss = []
        returns = []
        for r in r_list[::-1]:
            R = r + GAMMA * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        for log_prob, R in zip(self.log_probs_list, returns):
            policy_loss.append(-log_prob * R)
        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()
        del net.log_probs_list[:]

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self(state)
        m = Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        net.log_probs_list.append(log_prob)
        return action.clone().cpu().numpy()[0], log_prob
 
net = Softmax_policy_net().to(device)

optimizer = optim.RMSprop(net.parameters(), lr=1e-3)

total_reward_list = []

#def select_action(state, net):
#    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
#    probs = net(state)
#    m = Categorical(probs)
#    action = m.sample()
#    log_prob = m.log_prob(action)
#    net.log_probs_list.append(log_prob)
#    return action.clone().cpu().numpy()[0], log_prob
    

for epi in range(num_episode): 
    if epi == num_episode:
        state = 'playing'
    delta = 0

    EPSILON = max(eps_last, eps_ini* (1- epi*1./num_episode))
    ALPHA = max(alpha_last, alpha_ini* (1- epi*1./num_episode))
  
    done = False
    state, _ = env.reset() # 環境をリセット
    if render:
        env.render()

    s_list = [] # エピソードの状態履歴
    a_list = [] # エピソードの状態履歴
    r_list = [] # エピソードの状態履歴
    
    t = 0
    # エピソードを終端までプレイ
    while(done == False):
        if t >= 500:
            done = True
            break
        if render:
            env.render()
    
    
        action, log_prob = net.select_action(state)
        if  np.random.rand() < EPSILON:
            action = np.random.randint(NUM_ACTION)
        a_list.append(action)
        #print("action: %d" % action)
        state, reward, done, info, _ = env.step(action)
        r_list.append(reward)
        s_list.append(state)
    
        G_list = get_G_list(r_list, normalize=True)
        total_reward = np.sum(r_list)
        t += 1
    total_reward_list.append(total_reward)

    print(f"episode {epi},  total reward: {total_reward}, w[0,0]={net.linear1.weight[0,0].data.cpu().numpy()}")

    delta_list = []
    
    for i,(s,a,G, log_prob) in enumerate(zip(s_list, a_list, G_list, net.log_probs_list)):
       
        s = torch.from_numpy(s).float().unsqueeze(0).to(device)
        delta = torch.log(net(s))[0,a]
                                  
        #delta = (-log_prob *G).sum()
        #print(delta)
        delta_list.append((-log_prob * G).reshape(-1,1))
        #delta.backward()
        #net.update(G,i)
        #optimizer.step()
        #optimizer.zero_grad()

    
    optimizer.zero_grad()
    delta_all = torch.cat(delta_list).sum()
    delta_all.backward()
    optimizer.step()
    del net.log_probs_list[:]
    
    
    #net.finish_episode(r_list)
np.save("REINFORCE_vanilla.npy", np.array(total_reward_list))
