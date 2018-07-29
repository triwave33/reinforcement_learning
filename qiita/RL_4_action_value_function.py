### -*-coding:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt
import grid_world

       
# agentの生成
agent = grid_world.Agent([0,0])

ITER_NUM = 8 # 状態価値関数推定のためにt->ITER_NUMまで辿る

#
## 5*5全てのマスのに関して各行動の行動価値関数を計算
q_array = np.zeros((len(agent.ACTIONS), 5,5))
for index, action in enumerate(agent.ACTIONS):
    print("index: %d" % index)
    for i in range(5):
        for j in range(5):
            q_array[index, i,j] = agent.Q_pi([i,j],action, 1, 0, ITER_NUM)

# 結果をコンソールに表示
print("Qpi")
print(q_array)


## 結果をグラフィカルに表示

# 行動価値関数で最適な行動を検索
def if_true_color_red(val, else_color):
    if val:
        return 'r'
    else:
        return else_color
max_bool = q_array == np.max(q_array, axis=0)
max_color_k = np.vectorize(if_true_color_red)(max_bool,'k')
max_color_w = np.vectorize(if_true_color_red)(max_bool,'w')

# 表示用に小数点2桁に丸める
q_array_round = np.round(q_array, decimals=2)

# 行動価値関数を表示
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
plt.savefig('output_files/RL_4_q_pi.png')
plt.close()


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
        plt.arrow(i+ 0.5, j+0.5, 0.2, 0, width=0.01,head_width=0.15,\
            head_length=0.2,color=max_color_w[0,x,y])
        plt.arrow(i+ 0.5, j+0.5, 0, 0.2, width=0.01,head_width=0.15,\
            head_length=0.2, color=max_color_w[1,x,y])
        plt.arrow(i+ 0.5, j+0.5, -0.2, 0, width=0.01,head_width=0.15,\
            head_length=0.2, color=max_color_w[2,x,y])
        plt.arrow(i+ 0.5, j+0.5, 0, -0.2, width=0.01,head_width=0.15,\
            head_length=0.2, color=max_color_w[3,x,y])
plt.savefig('output_files/RL_4_opt_act.png')
plt.close()

# 状態価値関数を表示。状態関数は行動価値関数の行動に関しての期待をとる
ax = plt.gca()
plt.xlim(0,5)
plt.ylim(0,5)
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
v_array = np.mean(q_array, axis=0)
v_array_round = np.round(v_array, decimals=2)
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
plt.savefig('output_files/RL_4_v_pi.png')
plt.close()


