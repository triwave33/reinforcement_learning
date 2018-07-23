### -*-coding:utf-8-*-
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

class Agent():

    ## クラス変数定義

    # アクションと移動を対応させる辞書 
    actions = ['right', 'up', 'left', 'down']
    act_dict = {'right':np.array([0,1]), 'up':np.array([-1,0]),\
                'left':np.array([0,-1]), 'down':np.array([1,0])}
    
    num_action = len(actions) # 4

    # 上下左右全て等確率でランダムに移動する
    pi_dict1 = {'right':0.25, 'up':0.25,'left':0.25, 'down':0.25} 

    def __init__(self, array_or_list):
        # 入力はリストでもnp.arrayでも良い
        if type(array_or_list) == list:
            array = np.array(array_or_list)
        else:
            array = array_or_list
        assert (array[0] >=0 and array[0] < 5 and \
                array[1] >=0 and array[1] <5)
        self.pos = array

    # 現在位置を返す
    def get_pos(self):
        return self.pos

    # 現在位置をセットする
    def set_pos(self, array_or_list):
        if type(array_or_list) == list:
            array = np.array(array_or_list)
        else:
            array = array_or_list
        assert (array[0] >=0 and array[0] < 5 and \
                array[1] >=0 and array[1] <5)
        self.pos = array

    # 現在位置から移動
    def move(self, action):
        # 辞書を参照し、action名から移動量move_coordを取得
        move_coord = Agent.act_dict[action] 

        # A地点
        if (self.get_pos() == np.array([0,1])).all():
            pos_new = [4,1]
        # B地点
        elif (self.get_pos() == np.array([0,3])).all():
            pos_new = [2,3]
        else:
            pos_new = self.get_pos() + move_coord
        # グリッドの外には出られない
        pos_new[0] = np.clip(pos_new[0], 0, 4)
        pos_new[1] = np.clip(pos_new[1], 0, 4)
        self.set_pos(pos_new)

    # 現在位置から移動することによる報酬。この関数では移動自体は行わない
    def reward(self, state, action):
        # A地点
        if (state == np.array([0,1])).all():
            r = 10
            return r
        # B地点
        if (state == np.array([0,3])).all():
            r = 5
            return r
    
        # グリッドの境界にいて時に壁を抜けようとした時には罰則
        if (state[0] == 0 and action == 'up'):
            r = -1
        elif(state[0] == 4 and action == 'down'):
            r = -1
        elif(state[1] == 0 and action == 'left'):
            r = -1
        elif(state[1] == 4 and action == 'right'):
            r = -1
        # それ以外は報酬0
        else:
            r = 0
        return r
    
    def pi(self, state, action):
        # 変数としてstateを持っているが、実際にはstateには依存しない
        return Agent.pi_dict1[action]
    
    #def transition(self, state, action, next_state)
    
   
# agentの生成
agent = Agent([0,1])

GAMMA = 0.9
ACTIONS = ['right', 'up', 'left', 'down']
num_row = 5 
num_col = 5

# Vの初期化
V = np.zeros((5,5))

print("start iteration")

count = 0

N = 100
V_trend = np.zeros((N, num_row, num_col))

while(True):
    delta = 0
    for i in range(num_row):
        for j in range(num_col):
            #print("delta %f" % delta)
            v = V[i,j]
            tmp = 0
            for action in ACTIONS:
                agent.set_pos([i,j])
                s = agent.get_pos()
                agent.move(action)
                s_dash = agent.get_pos()
                tmp += agent.pi(s, action) * 1.0 *\
                        (agent.reward(s,action) + GAMMA * V[s_dash[0], s_dash[1]])
                #print("i: %d, j: %d, s:%s, a:%5s, s';%s" %(i,j, str(s), action,  str(s_dash)))
            V[i,j] = tmp
            delta = max(delta, abs(v - V[i,j]))
    #print("count: %d, delta: %f, abs: %f" % (count, delta, abs(v-V[i,j])))
    count += 1
    V_trend[count, :,:] = V
    if delta < 1.E-5:
        break
    
print(V)

# Vの収束をプロット
for i in range(num_row):
    for j in range(num_col):
        plt.subplot(5,5, i*5+j+1)
        plt.plot(V_trend[:count,i,j])

plt.show()

