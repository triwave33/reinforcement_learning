# reinforcement_learning

日本語の説明は作成中。(Japanese descripsion is now being constructed...)

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
