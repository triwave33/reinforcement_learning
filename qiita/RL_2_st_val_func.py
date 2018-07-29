### -*-coding:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt
import grid_world
   
# agentの生成
agent = grid_world.Agent([0,0])

ITER_NUM = 8 # 状態価値関数推定のためにt->ITER_NUMまで辿る

agent.V_pi([0,1], 1, 0, ITER_NUM)

print("状態価値関数（Vpi")
print("%dステップ先までを計算" % ITER_NUM)

# 5*5全てのマスの状態価値関数を計算
v_array_trend = np.zeros((ITER_NUM, 5,5)) #V_pi格納用
for iter_num in range(ITER_NUM):
    for i in range(5):
        for j in range(5):
            v_array_trend[iter_num,i,j] = agent.V_pi([i,j],1,0,iter_num+1)

# 最終結果をコンソールに表示
print("v_array after %d iteration" % ITER_NUM)
print(v_array_trend[-1,:,:])

# 5x5マスにプロット
for i in range(5):
    for j in range(5):
        plt.subplot(5,5,5*i+j+1)
        plt.plot(v_array_trend[:,i,j])
plt.show()


