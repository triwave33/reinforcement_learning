### -*-coding:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt



class Agent():

    ## クラス変数定義
    # アクションと移動を対応させる辞書 
    GAMMA = 0.9
    ACTIONS = ['right', 'up', 'left', 'down']
    act_dict = {'right':np.array([0,1]), 'up':np.array([-1,0]),\
                'left':np.array([0,-1]), 'down':np.array([1,0])}
    num_action = len(ACTIONS) # 4
    num_row = 4
    num_col = 12
    
    
    # 上下左右全て等確率でランダムに移動する
    pi_dict1 = {'right':0.25, 'up':0.25,'left':0.25, 'down':0.25} 
    
    def __init__(self):
        self.pos = np.array([3,0])
    
    # 現在位置を返す
    def get_pos(self):
        return self.pos
    
    # 現在位置をセットする
    def set_pos(self, array_or_list):
        if type(array_or_list) == list:
            array = np.array(array_or_list)
        else:
            array = array_or_list
        assert (array[0] >=0 and array[0] < 4 and \
                array[1] >=0 and array[1] <12)
        self.pos = array
    
    # 現在位置から移動
    def move(self, action):
        terminate = False
        # 辞書を参照し、action名から移動量move_coordを取得
        move_coord = Agent.act_dict[action] 
        reward =  -1
    
        pos_new = self.get_pos() + move_coord
        pos_new[0] = np.clip(pos_new[0], 0,3)
        pos_new[1] = np.clip(pos_new[1], 0,11)

        #print("pos_new: %d, %d" % (pos_new[0], pos_new[1]))
        
        # criff
        if ((pos_new[0] == 3)& (pos_new[1] >=1) & (pos_new[1] <=10)):
            # print("In the criff")
            pos_new = np.array([3,0])
            terminate = True
            reward = -100

        if (pos_new == np.array([3,11])).all():
            pos_new = np.array([3,0])
            terminate = True
            #reward = 1000
            reward = 100

        self.set_pos(pos_new)
        return pos_new, reward, terminate

    
    def pi(self, state, action):
        # 変数としてstateを持っているが、実際にはstateには依存しない
        return Agent.pi_dict1[action]
    
    
# 最適行動に赤色のラベル、他には指定したカラーラベルをつける
def if_true_color_red(val, else_color):
    if val:
        return 'r'
    else:
        return else_color


def Q_value_plot(Q):
    max_bool = Q == np.max(Q, axis=0)
    max_color_k = np.vectorize(if_true_color_red)(max_bool,'k')
    max_color_w = np.vectorize(if_true_color_red)(max_bool,'w')
    q_array_round = np.round(Q, decimals=2)
    # 行動価値関数を表示
    ax = plt.gca()
    plt.xlim(0,Agent.num_col)
    plt.ylim(0,Agent.num_row)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    
    for i in range(Agent.num_col):
        for j in range(Agent.num_row):
            # rect
            rect = plt.Rectangle(xy =(i,j) , width=1, height=1, fill=False)
            ax.add_patch(rect)
            # diag
            diag = plt.Line2D(xdata=(i,i+1), ydata=(j,j+1),color='k',linewidth=.5)
            ax.add_line(diag)
            diag = plt.Line2D(xdata=(i,i+1), ydata=(j+1,j),color='k',linewidth=.5)
            ax.add_line(diag)
            # 座標のインデックスの調整
            x = -j-1 
            y = i
            # text
            plt.text(i+ 0.65, j+0.45, "%s" % (str(q_array_round[0,x,y])), color=max_color_k[0,x,y])
            plt.text(i+ 0.4, j+0.8, "%s" % (str(q_array_round[1,x,y])), color=max_color_k[1,x,y])
    
            plt.text(i+ 0.025, j+0.45, "%s" % (str(q_array_round[2,x,y])), color=max_color_k[2,x,y])
    
            plt.text(i+ 0.4, j+0.1, "%s" % (str(q_array_round[3,x,y])), color=max_color_k[3,x,y])


def Q_arrow_plot(Q):
    max_bool = Q == np.max(Q, axis=0)
    max_color_k = np.vectorize(if_true_color_red)(max_bool,'k')
    max_color_w = np.vectorize(if_true_color_red)(max_bool,'w')
    q_array_round = np.round(Q, decimals=2)
    # 最適行動を矢印で表示
    ax = plt.gca()
    plt.xlim(0,Agent.num_col)
    plt.ylim(0,Agent.num_row) 
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    
    for i in range(Agent.num_col):
        for j in range(Agent.num_row):
            # rect
            rect = plt.Rectangle(xy =(i,j) , width=1, height=1, fill=False)
            ax.add_patch(rect)
            # 座標のインデックスの調整
            x = -j-1 
            y = i
            # arrow
            plt.arrow(i+ 0.5, j+0.5, 0.2, 0, width=0.01,head_width=0.15,\
                head_length=0.2,color=max_color_w[0,x,y])
            plt.arrow(i+ 0.5, j+0.5, 0, 0.2, width=0.01,head_width=0.15,\
                head_length=0.2, color=max_color_w[1,x,y])
            plt.arrow(i+ 0.5, j+0.5, -0.2, 0, width=0.01,head_width=0.15,\
                head_length=0.2, color=max_color_w[2,x,y])
            plt.arrow(i+ 0.5, j+0.5, 0, -0.2, width=0.01,head_width=0.15,\
                head_length=0.2, color=max_color_w[3,x,y])

def empty_plot(Q):
    max_bool = Q == np.max(Q, axis=0)
    max_color_k = np.vectorize(if_true_color_red)(max_bool,'k')
    max_color_w = np.vectorize(if_true_color_red)(max_bool,'w')
    q_array_round = np.round(Q, decimals=2)
    # 最適行動を矢印で表示
    ax = plt.gca()
    plt.xlim(0,Agent.num_col)
    plt.ylim(0,Agent.num_row) 
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    
    for i in range(Agent.num_col):
        for j in range(Agent.num_row):
            # rect
            rect = plt.Rectangle(xy =(i,j) , width=1, height=1, fill=False)
            ax.add_patch(rect)

def V_value_plot(V):
    # 状態価値関数を表示
    ax = plt.gca()
    plt.xlim(0,5)
    plt.ylim(0,5)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    v_array_round = np.round(V, decimals=2)
    for i in range(5):
        for j in range(5):
            # rect
            rect = plt.Rectangle(xy =(i,j) , width=1, height=1, fill=False)
            ax.add_patch(rect)
            # 座標のインデックスの調整
            x = -j-1 
            y = i
            # text
            plt.text(i+ 0.4, j+0.5, "%s" % (str(v_array_round[x,y])))

def pi_arrow_plot(pi):
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

agent = Agent()
s, r, t = agent.move('up')
print s
