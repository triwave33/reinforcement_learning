### -*-coding:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt

# 状態価値関数を表示。状態関数は行動価値関数の行動に関しての期待をとる
def plot(V, filename):
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
    plt.savefig('output_files/'+filename)
    plt.close()
    

