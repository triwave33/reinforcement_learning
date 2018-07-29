### -*-coding:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt
import grid_world

# agentの生成
agent = grid_world.Agent([0,0])

num_row = 5 
num_col = 5

# Vの初期化
V = np.zeros((num_row, num_col))
# piの初期化
pi = np.random.randint(0,len(agent.ACTIONS),(num_row,num_col)) # 確定的な方策

print("")
print("pi initial")
print(pi)

print("start iteration")

N = 1000
V_trend = np.zeros((N, num_row, num_col))

count = 0
while(True):
    delta = 0
    for i in range(num_row):
        for j in range(num_col):
            tmp = np.zeros(len(agent.ACTIONS))
            v = V[i,j]
            for index,action in enumerate(agent.ACTIONS):
            #print("delta %f" % delta)
                agent.set_pos([i,j])
                s = agent.get_pos()
                agent.move(action)
                s_dash = agent.get_pos()
                tmp[index] =  (agent.reward(s,action) + agent.GAMMA * V[s_dash[0], s_dash[1]])
            V[i,j] = max(tmp)
            V_trend[count, :,:] = V
            delta = max(delta, abs(v - V[i,j]))
    if delta < 1.E-5:
        break
    count += 1

for i in range(num_row):
    for j in range(num_col):
        tmp = np.zeros(len(agent.ACTIONS))
        for index, action in enumerate(agent.ACTIONS):
            agent.set_pos([i,j])
            s = agent.get_pos()
            agent.move(action)
            s_dash = agent.get_pos()
            tmp[index] =  (agent.reward(s,action) + agent.GAMMA * V[s_dash[0], s_dash[1]])
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
plt.savefig('output_files/RL_6_2_opt_act.png')
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
plt.savefig('output_files/RL_6_2_v_pi.png')
plt.close()


