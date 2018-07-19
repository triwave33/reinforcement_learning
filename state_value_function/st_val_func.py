### -*-coding:utf-8-*-
import numpy as np
from itertools import product


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
    
    def V_pi(self, state, n, out, iter_num):
        #print('\nnow entering V_pi at n=%d, state=%s' %(n, str(state)))
        # state:関数呼び出し時の状態
        # n:再帰関数の呼び出し回数。関数実行時は1を指定
        # out:返り値用の変数。関数実行時は0を指定
    
        if n==iter_num:    # 終端状態
            for i, action in enumerate(Agent.actions):
                out += self.pi(state, action) * self.reward(state,action)
                #print("terminal condition")
            return out
        else:
            for i, action in enumerate(ACTIONS):
                #print("action: %s" % action)
                out += self.pi(state, action)  * self.reward(self.get_pos(),action) # 報酬
                #print("before move, pos:%s" % str(self.get_pos()))
                self.move(action) # 移動してself.get_pos()の値が更新
                #print("after move, pos:%s" % str(self.get_pos()))
    
                ## 価値関数を再帰呼び出し
                # state変数には動いた先の位置、つまりself.get_pos()を使用
                out +=  self.pi(self.get_pos(), action) * \
                        self.V_pi(self.get_pos(), n+1, 0,  iter_num) * GAMMA
                agent.set_pos(state) #  再帰先から戻ったらエージェントを元の地点に初期化
                #print("agent pos set to %s, at n:%d" % (str(state), n))
            return out

   
# agentの生成
agent = Agent([0,1])

ITER_NUM = 4 # 状態価値関数推定のためにt->ITER_NUMまで辿る
GAMMA = 0.9
ACTIONS = ['right', 'up', 'left', 'down']

agent.V_pi([0,1], 1, 0, ITER_NUM)

print("状態価値関数（Vpi")
print("%dステップ先までを計算" % ITER_NUM)

# 5*5全てのマスの状態価値関数を計算
v_array = np.zeros((5,5))
for i in range(5):
    for j in range(5):
        v_array[i,j] = agent.V_pi([i,j], 1, 0, ITER_NUM)

print("Vpi")

print(v_array)






