using OpenAIGym
using Printf
using ProgressBars

#env_name = :MountainCar
env_name = :MountainCar
env_name = :CartPole

env = GymEnv(env_name, :v0)


if env_name == :MountainCar
    min_list = [-1.2, -0.04]::Array{Float64, 1}
    max_list = [0.6,0.07]::Array{Float64, 1}
    const num_state = 2::Int64
    const num_action =3::Int64
elseif env_name ==:CartPole
    min_list = [-4.8, -3.4, -0.4, -3.4]::Array{Float64, 1}
    max_list = [4.8, 3.4, 0.4, 3.4]::Array{Float64, 1}
    const num_state = 4::Int64
    const num_action =2::Int64
end

# State space
const N = 32::Int64

 
 
#ACTIONS = range(0,num_action-1, step=1)  # should be 0-based
const ALPHA = 0.1::Float64
const ALPHA_INI = 0.1::Float64
const ALPHA_LAST = 0.1::Float64
const GAMMA = 0.99::Float64
const EPS_INI = 0.5::Float64
const EPS_LAST = 0.0::Float64
const num_episode = 50000::Int64 # sampling num
const render = false
const ex_factor = 1.0::Float64 # 

# initialize Q
Q = zeros(num_action, repeat([N], num_state)...)
Q_10trial = zeros((10,  Int(round(num_episode*ex_factor))))

# visit number
visit_array = zeros((num_action, N, N))

reward_array = zeros(Int(round(num_episode * ex_factor)))

# initialize probability
p = rand(num_action)::Array{Float64,1}
p = p / sum(rand(3)) # standardization

function select_action(Q::Array{Float64,num_state+1}, s::Array{Int64,1}, eps::Float64)
    # select action based on e-greedy
    greedy = argmax(Q[:, s...])
    p = [(if i == greedy 1-eps + eps/num_action
        else eps/num_action end) for i in 1:num_action]
    return sample(1:num_action, Weights(p))
end

function digitize_state(observation::Array{Float64,1}, min_list::Array{Float64,1}, max_list::Array{Float64,1}, N::Int64, num_state::Int64)
   s = [Int(round(searchsortedfirst(range(min_list[i], max_list[i], length=N-1), observation[i]))) for i in 1:num_state]
   return s
end

 


function main()
    reward_list = []
    for epi in tqdm(1:(Int(round(num_episode * ex_factor))))
        # greedy decrease to 0
        EPSILON = max(EPS_LAST, EPS_INI * (1 - epi*(1.)/num_episode))
        ALPHA = max(ALPHA_LAST, ALPHA_INI * (1 - epi*(1.)/num_episode))
    
        # initialize s
        done = false
        observation = env.pyreset()
        # discretize
        s = digitize_state(observation, min_list, max_list, N, num_state)
    
        global tmp = 0
        global c = 0
    
        # run episode until terminal
        while(done==false)
            if render
                env.render()
            end
    
            # action selecton by e-greedy
            a = select_action(Q, s, EPSILON)
    
            visit_array[a, s[1], s[2]] += 1
    
    
            observation, reward, done, info = env.pystep(a-1) # -1 for python consistensy
    
            # digitize state
            s_dash = digitize_state(observation, min_list, max_list, N, num_state)
    
   
            global tmp += reward
    
            # select action by argmax(Q_learning)
            Q_dash = maximum(Q[:,s_dash...])
    
            # update Q
            Q[a, s...] += ALPHA * (reward + GAMMA*(Q_dash)
                                  - Q[a, s...])
            s = s_dash
    
            c += 1
    
            push!(reward_list, tmp)
        end
          #@printf("N: %d, epi: %d, eps: %.3f, reward: %3d\n" , N, epi, EPSILON, tmp)
    end    
end

@time main()


