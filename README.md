# reinforcement_learning
(日本語による説明は一番下にあります)



<br>

<img width="400" alt="IMG_CD5BE22E35D2-1.jpeg" src="https://camo.qiitausercontent.com/63d0b9f26bdbf8cb60ff52527aab278c13a795ac/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e616d617a6f6e6177732e636f6d2f302f3233333230382f33346163363464642d383366362d333065332d613464612d3534626432323235306564332e676966">
Value function approximation using Linear kernel functions.

<br>
<br>
<br>

## Overview

This repository is an archive of my learning for reinforcement learning according to a great book "Reinforce ment learning" by Sutton, S.S. and Andrew, G.B. (Japanese edition).

Some algorithms in the book are implemented and examples described there are solved from full scrach (not using any RL libraries).

Detailed description for each script is descripted in my blog (https://qiita.com/triwave33 BUT sorry in Japanese)




## Grid World
<img width="597" alt="IMG_CD5BE22E35D2-1.jpeg" src="https://qiita-image-store.s3.amazonaws.com/0/233208/74500b1a-3802-0533-099b-2348e5539dc6.jpeg">

Some scripts in this repository are treating grid-worl problems in Sutton book pp. 77-78 (but sorry again, this information is for Japanese edition).
The essence is below.

- Agent can take action (i.e. move) toward left, up, right and down directions.
- Agent cannot move outside the grid and stays even if taking some actions. 
- If Agent stands on the place "A" or "B", he/she gets 10 or 5 points AFTER taking any action and warps to A' or B', respectively

### contents of grid-world problem

- RL_2_st_val_func.py: estimating value function  by recursive calculation 
- RL_4_action_value_function.py: estimating action value function by recursive calculation 
- RL_5_iterative_policy_evaluation.py: estimating state value function by iterative bootstrapping (chapter 4.1)
- RL_6_1_iterative_policy_improvement.py (chapter 4.3)
- RL_7_first_visit_MC.py: (chapter 5.1)

## OpenAI Gym 
- RL_8_cartpole_montecarlo.py: solvimg "CartPole" problem from OpenAI Gym environment (chapter 5.4)


# 日本語による説明
qiitaのサイトのコード置き場です。ファイル名の番号とqiitaの記事の番号が対応しています。

[今さら聞けない強化学習（1）：状態価値関数とBellman方程式](https://qiita.com/triwave33/items/5e13e03d4d76b71bc802) 

[今さら聞けない強化学習（2）：状態価値関数の実装](https://qiita.com/triwave33/items/3bad9f35d213a315ce78) 

[今さら聞けない強化学習（3）：行動価値関数とBellman方程式](https://qiita.com/triwave33/items/8966890701169f8cad47) 

[今さら聞けない強化学習（4）：行動価値関数の実装](https://qiita.com/triwave33/items/669a975b74461559addc) 

[今さら聞けない強化学習（5）：状態価値関数近似と方策評価](https://qiita.com/triwave33/items/bed0fd7a2b56ee8e7c29) 

[今さら聞けない強化学習（6）：反復法による最適方策](https://qiita.com/triwave33/items/59768d14da38f50fb76c) 

[今さら聞けない強化学習（7）：モンテカルロ法で価値推定](https://qiita.com/triwave33/items/0c8833e6b899c26b208e) 

[今さら聞けない強化学習（8）: モンテカルロ法でOpenAI GymのCartpoleを学習](https://qiita.com/triwave33/items/1b9c87089b2fce0dd481) 

[今さら聞けない強化学習（9）:TD法の導出](https://qiita.com/triwave33/items/277210c7be4e47c28565) 

[今さら聞けない強化学習(10): SarsaとQ学習の違い](https://qiita.com/triwave33/items/cae48e492769852aa9f1)

[今さら聞けない強化学習(11) 線形関数による価値関数近似](https://qiita.com/triwave33/items/78780ec37babf154137d)
