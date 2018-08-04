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
#方策を矢印で表示
grid_world.pi_arrow_plot(pi)
#状態価値関数を表示
grid_world.V_value_plot(V)
