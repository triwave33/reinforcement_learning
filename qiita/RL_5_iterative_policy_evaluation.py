### -*-coding:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt
import grid_world

# agentの生成
agent = grid_world.Agent([0,0])

num_row = 5 
num_col = 5

# Vの初期化
V = np.zeros((num_row,num_col))

print("start iteration")

count = 0

N = 100
V_trend = np.zeros((N, num_row, num_col)) # 今回反復回数はN(100)で十分

while(True):
    delta = 0
    for i in range(num_row):
        for j in range(num_col):
            #print("delta %f" % delta)
            v = V[i,j]
            tmp = 0
            for action in agent.ACTIONS:
                agent.set_pos([i,j]) # 移動前の状態に初期化
                s = agent.get_pos()
                agent.move(action) # 移動
                s_dash = agent.get_pos() # 移動後の状態
                tmp += agent.pi(s, action) * 1.0 *\
                        (agent.reward(s,action) + agent.GAMMA * V[s_dash[0], s_dash[1]])
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

