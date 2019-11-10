using OpenAIGym
using Printf

env = GymEnv(:MountainCar, :v0)



# State space
N = 32
min_list = env.pyenv.observation_space.low
min_list = env.pyenv.observation_space.low

min_list = [-1.2, -0.04]
max_list = [0.6,0.07]

ACTIONS = [0,1,2]
ALPHA = 0.1
ALPHA_INI = 0.1
ALPHA_LAST = 0.1
GAMMA = 1.
EPS_INI = 0.5
EPS_LAST = 0.2
num_state = 2
num_action =3
num_episode = 50000 # sampling num
render = false
ex_factor = 1.0 # 

# initialize Q
Q = zeros((num_action, N, N))
Q_10trial = zeros((10,  Int(round(num_episode*ex_factor))))

# visit number
visit_array = zeros((num_action, N, N))

reward_array = zeros(Int(round(num_episode * ex_factor)))

# initialize probability
p = rand(num_action)
p = p / sum(rand(3)) # standardization

function select_action(Q, s, eps)
    # select action based on e-greedy
    greedy = argmax(Q[:, s[1], s[2]])
    p = [(if i == greedy 1-eps + eps/length(ACTIONS) 
        else eps/length(ACTIONS) end) for i in 1:length(ACTIONS)]
    return sample(1:num_action, Weights(p))
end

function digitize_state(observation, min_list, max_list, N, num_state)
   s = [Int(round(searchsortedfirst(range(min_list[i], max_list[i], length=N-1), observation[i]))) for i in 1:num_state]
end

 


reward_list = []
min_pos = min_list[1]
max_pos = max_list[1]

for epi in 1:(Int(round(num_episode * ex_factor)))
    # greedy decrease to 0
    EPSILON = max(EPS_LAST, EPS_INI * (1 - epi*(1.)/num_episode))
    ALPHA = max(ALPHA_LAST, ALPHA_INI * (1 - epi*(1.)/num_episode))

    # initialize s
    done = false
    observation = env.pyreset()
    # discretize
    s = [Int(round(searchsortedfirst(range(min_list[i], max_list[i], length=N-1), observation[i]))) for i in 1:num_state]

    global tmp = 0
    global count = 0

    # run episode until terminal
    while(done==false)
        if render
            env.render()
        end

        # action selecton by e-greedy
        a = select_action(Q, s, EPSILON)

    e   visit_array[a, s[1], s[2]] += 1


        observation, reward, done, info = env.pystep(a-1) # -1 for python consistensy

        # digitize state
        s_dash = [Int(round(searchsortedfirst(range(min_list[i], max_list[i], length=N-1), observation[i]))) for i in 1:num_state]

        #if min_pos > observation[1]
        #    min_pos = observation[1]
        #end
        #if max_pos < observation[1]
        #    max_pos = observation[1]
        #end

        global tmp += reward

        # select action by argmax(Q_learning)
        Q_dash = maximum(Q[:,s_dash[1], s_dash[2]])

        # update Q
        Q[a, s[1], s[2]] += ALPHA * (reward + GAMMA*(Q_dash)
                              - Q[a, s[1], s[2]])
        s = s_dash

        count += 1

        push!(reward_list, tmp)
    end
    @printf("N: %d, epi: %d, eps: %.3f, reward: %3d\n" , N, epi, EPSILON, tmp)

end
