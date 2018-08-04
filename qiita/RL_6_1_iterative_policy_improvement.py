### -*-coding:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt
import grid_world

# agentの生成
agent = grid_world.Agent([0,0])


print("start iteration")
N = 1000
num_row = 5 
num_col = 5

# 1.初期化
# Vの初期化
V = np.zeros((num_row, num_col))
# piの初期化
pi = np.random.randint(0,len(agent.ACTIONS),(num_row,num_col)) # 決定的な方策
V_trend = np.zeros((N, num_row, num_col)) # V格納用テーブル
pi_trend = np.zeros((N, num_row, num_col)) # pi格納用テーブル



entire_count=0
policy_stable = False
while(policy_stable == False):
    # 2.方策評価
    print('entire count %d: ' % entire_count)
    count = 0
    while(True):
        delta = 0
        # 状態空間をスキャン
        for i in range(num_row):
            for j in range(num_col):
                #print("delta %f" % delta)
                v = V[i,j]
                # 方策に従い行動を決定
                action = agent.ACTIONS[pi[i,j]]
                agent.set_pos([i,j])
                s = agent.get_pos()
                # 行動を実行し次の状態に遷移
                agent.move(action)
                s_dash = agent.get_pos()
                tmp =  (agent.reward(s,action) + agent.GAMMA * V[s_dash[0], s_dash[1]])
                V[i,j] = tmp
                delta = max(delta, abs(v - V[i,j]))
        count += 1
        if delta < 1.E-5:
            break
    
    V_trend[entire_count, :,:] = V
    # 3.方策改善
    b = pi.copy()
    # 状態空間をスキャン
    for i in range(num_row):
        for j in range(num_col):
            tmp = np.zeros(len(agent.ACTIONS))
            # 最適な行動を決定するため取り得る行動についてスキャン
            for index, action in enumerate(agent.ACTIONS):
                agent.set_pos([i,j])
                s = agent.get_pos()
                # 行動を実行、次の状態に遷移
                agent.move(action)
                s_dash = agent.get_pos()
                tmp[index] = agent.reward(s,action) + agent.GAMMA*V[s_dash[0], s_dash[1]]
            # 最適な行動を決定
            pi[i,j] = np.argmax(tmp)
    pi_trend[:entire_count,:,:] = pi
    if(np.all(b==pi)):
        policy_stable=True
        print("policy_stable")
        break
    entire_count += 1

V_trend= V_trend[:entire_count,:,:]
pi_trend= pi_trend[:entire_count,:,:]
       


## 結果をグラフィカルに表示
# 方策piを矢印で表示
grid_world.pi_arrow_plot(pi)
# 状態価値関数を表示
grid_world.V_value_plot(V)

