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
grid_world.Q_value_plot(q_array)
grid_world.Q_arrow_plot(q_array)

v_array = np.max(q_array, axis=0)
grid_world.V_value_plot(v_array)
