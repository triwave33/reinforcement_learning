
import gym
import numpy as np
from torch import nn
import torch
from torch.distributions import Categorical
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_for_loop = False
env = gym.make('CartPole-v0')
num_episode = 1000

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


class SoftmaxPolicy(nn.Module):
    def __init__(self):
        super(SoftmaxPolicy, self).__init__()
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
        policy_optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        policy_optimizer.step()
        del self.log_probs_list[:]

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self(state)
        m = Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        self.log_probs_list.append(log_prob)
        return action.clone().cpu().numpy()[0], log_prob


class StateValue(nn.Module):
    def __init__(self):
        super(StateValue, self).__init__()
        self.num_action = NUM_ACTION
        self.feature_dimension = FEATURE_DIM
        self.linear1 = nn.Linear(in_features= self.feature_dimension, 
                                out_features= 128,
                                bias = True)
        self.linear2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(p=0.6)
        self.log_probs_list = []


    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x =  self.linear2(x)
        return F.relu(x)

    def estimate_value_from_state(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        value = self(state)
        return value

    def change_requires_grad(self, b):
        for param in self.parameters():
            param.requires_grad = b





policy = SoftmaxPolicy().to(device)
critic = StateValue().to(device)

policy_optimizer = optim.RMSprop(policy.parameters(), lr=1e-3)
critic_optimizer = optim.RMSprop(critic.parameters(), lr=1e-3)
double_optimizer = optim.RMSprop(list(critic.parameters()) + list(policy.parameters()),  lr=1e-3)

total_reward_list = []

   

step_count = 0
b = 0
for epi in range(num_episode): 
    if epi == num_episode:
        state = 'playing'
    delta = 0

  
    done = False
    state, _ = env.reset() # 環境をリセット
    if render:
        env.render()

    s_list = [] # エピソードの状態履歴
    a_list = [] # エピソードの状態履歴
    r_list = [] # エピソードの状態履歴
    v_list = []

    t = 0
    I = 1
    # エピソードを終端までプレイ
    while(done == False):
        if t >= 500:
            done = True
            break


        if render:
            env.render()
    
        v = critic.estimate_value_from_state(state) 
        action, log_prob = policy.select_action(state)
        a_list.append(action)
        #print("action: %d" % action)
        # ACTION
        state_dash, reward, done, info, _ = env.step(action)

        v_next = critic.estimate_value_from_state(state_dash)
        delta = (reward + GAMMA * v_next - v).detach()
        delta_v = -delta * v
        I_delta_lnPi = I * -delta * log_prob

        v_list.append(v)
        r_list.append(reward)
        s_list.append(state)
        G_list = get_G_list(r_list, normalize=True)
        total_reward = np.sum(r_list)
        t += 1

        critic_optimizer.zero_grad()
        delta_v.backward(retain_graph=True)
        critic_optimizer.step()
        # update policy 
        policy_optimizer.zero_grad()
        I_delta_lnPi.backward()
        policy_optimizer.step()
        del policy.log_probs_list[:]
        t +=1
        I *= GAMMA
        state = state_dash
 
    total_reward_list.append(total_reward)

    print(f"episode {epi},  total reward: {total_reward}, w[0,0]={policy.linear1.weight[0,0].data.cpu().numpy()}")


    # update critic
   


np.save("actor_critic.npy", np.array(total_reward_list))
