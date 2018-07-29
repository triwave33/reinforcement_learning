### -*-coding:utf-8-*-
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

class Gambler():

    ## クラス変数定義
    # アクションと移動を対応させる辞書 
    actions = ['right', 'up', 'left', 'down']
    act_dict = {'right':np.array([0,1]), 'up':np.array([-1,0]),\
                'left':np.array([0,-1]), 'down':np.array([1,0])}
    
    num_action = len(actions) # 4
    
    # 表が出る確率
    p = 0.4

    def __init__(self, capital):
        assert (capital >=0 and capital <=100)
        self.capital = capital
        self.terminate = False

    # 現在の資本金を返す
    def get_capital(self):
        return self.capital

    # 資本金をセットする
    def set_capital(self, capital):
        assert (capital >=0 and capital <=100)
        self.capital = capital

    # 賭ける 
    def bet(self, money):
        s = self.get_capital()
        assert(money >=1 and money <= min(s, 100-s))
        if np.random.rand() <=0.4:
            self.set_capital(s+money)
        else:
            self.set_capital(s-money)

        s = self.get_capital()
        if (s==0):
            self.terminate = True
            return 0
        elif (s==100):
            self.terminate = True
            return 1
        else:
            return 1

    def P(s, 
  
   
# agentの生成
gambler = Gambler(1)

GAMMA = 1
N = 100 
# Vの初期化
V = np.zeros((N+1)) # 0-100
pi = np.zeros((N+1)) # 0-99
V[N] = 1 

# piの初期化
for i in range(1,N):
    pi[i] = np.random.randint(1,(min(i, 100-i)+1))

pi = np.random.randint(0,4,(5,5)) # 確定的な方策

print("")
print("pi initial")
print(pi)

print("start iteration")


count = 0
while(True):
    delta = 0
    for i in range(1,N):
        tmp = np.zeros(len(ACTIONS))
        v = V[i]
        s = gambler.get_capital()
        bet_range = range(1,min(s,100-s)+1)
        for index in bet_range:
        #print("delta %f" % delta)
            agent.set_pos([i,j])
            s = agent.get_pos()
            agent.move(action)
            s_dash = agent.get_pos()
            tmp[index] =  (agent.reward(s,action) + GAMMA * V[s_dash[0], s_dash[1]])
        V[i,j] = max(tmp)
        V_trend[count, :,:] = V
        delta = max(delta, abs(v - V[i,j]))
    if delta < 1.E-5:
        break
    count += 1

for i in range(num_row):
    for j in range(num_col):
        tmp = np.zeros(len(ACTIONS))
        for index, action in enumerate(ACTIONS):
            agent.set_pos([i,j])
            s = agent.get_pos()
            agent.move(action)
            s_dash = agent.get_pos()
            tmp[index] =  (agent.reward(s,action) + GAMMA * V[s_dash[0], s_dash[1]])
        pi[i,j] = np.argmax(tmp)

V_trend= V_trend[:count,:,:]
       


## 結果をグラフィカルに表示

# 方策piから最適な行動を検索
def if_true_color_red(val, else_color):
    if val:
        return 'r'
    else:
        return else_color
max_color_k = np.vectorize(if_true_color_red)(pi,'k')
max_color_w = np.vectorize(if_true_color_red)(pi,'w')


# 最適行動を矢印で表示
ax = plt.gca()
plt.xlim(0,5)
plt.ylim(0,5) 
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])

for i in range(5):
    for j in range(5):
        # rect
        rect = plt.Rectangle(xy =(i,j) , width=1, height=1, fill=False)
        ax.add_patch(rect)
        # 座標のインデックスの調整
        x = -j-1 
        y = i
        # arrow
        if pi[x,y] ==0:
            plt.arrow(i+ 0.5, j+0.5, 0.2, 0, width=0.01,head_width=0.15,\
                head_length=0.2,color='r')
        elif pi[x,y] ==1:
            plt.arrow(i+ 0.5, j+0.5, 0, 0.2, width=0.01,head_width=0.15,\
                head_length=0.2, color='r')
        elif pi[x,y] ==2:
            plt.arrow(i+ 0.5, j+0.5, -0.2, 0, width=0.01,head_width=0.15,\
                head_length=0.2, color='r')
        elif pi[x,y] ==3:
            plt.arrow(i+ 0.5, j+0.5, 0, -0.2, width=0.01,head_width=0.15,\
                head_length=0.2, color='r')
plt.savefig('opt_act.png')
plt.close()

# 状態価値関数を表示。状態関数は行動価値関数の行動に関しての期待をとる
ax = plt.gca()
plt.xlim(0,5)
plt.ylim(0,5)
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
V_round = np.round(V, decimals=2)
for i in range(5):
    for j in range(5):
        # rect
        rect = plt.Rectangle(xy =(i,j) , width=1, height=1, fill=False)
        ax.add_patch(rect)
        # 座標のインデックスの調整
        x = -j-1 
        y = i
        # text
        plt.text(i+ 0.4, j+0.5, "%s" % (str(V_round[x,y])))
plt.savefig('v_pi.png')
plt.close()


