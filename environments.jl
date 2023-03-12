using DelimitedFiles
using Distributions
using StatsBase
using PyCall

# An implementation of Environment REQUIRES multiple dispatch implementations of:
# step!, reset!, state, action_space, observation_space
abstract type Environment end
abstract type Space end

struct DiscreteSpace
    size::Array{Int32}
end

struct ContinuousSpace
    size::Array{Tuple{Float32, Float32}}
end

# Racetrack environment definition (from Sutton chapter 5 page 112)
mutable struct Racetrack <: Environment
    track::Matrix{Int32}
    width::Int32
    height::Int32
    action_space::DiscreteSpace
    observation_space::DiscreteSpace
    state::Array{Int32}
    start_coords::Tuple{Int32, Int32}
    action_mapping::Array{Tuple{Int32, Int32}}
end

function Racetrack(file::String)
    track = []
    open(file) do rt
        track = readdlm(rt, ',', Int8)
    end

    track = reverse(track; dims=1)
    width = size(track)[1]
    height = size(track)[2]

    start_coords = (0,0)
    for x in 1:width
        for y in 1:height
            v = track[x,y]
            if v == 3
                start_coords = (x, y)
            end
            print(v)
        end
        print("\n")
    end

    action_mapping = [(0,0), (-1, 0), (1, 0), (0, -1), (0, 1), (1, 1), (-1, -1)]
    return Racetrack(track, width, height,
                     DiscreteSpace([7]), DiscreteSpace([width, height, 6, 6]),
                     [start_coords[1], start_coords[2], 1, 1], start_coords, action_mapping)
end

function state(env::Racetrack)
    return env.state
end

function action_space(env::Racetrack)
    return env.action_space
end

function observation_space(env::Racetrack)
    return env.observation_space
end

function reset!(env::Racetrack)
    env.state = [env.start_coords[1], env.start_coords[2], 1, 2]
end

function step!(env::Racetrack, action)
    x, y, vx, vy = env.state
    nx = x + vx - 1
    ny = y + vy - 1
    nvx = vx + env.action_mapping[action][1]
    nvy = vy + env.action_mapping[action][2]
    reward = -1

    # can't go off the track
    if nx > env.width || ny > env.height || nx < 1 || ny < 1
        nx = env.start_coords[1]
        ny = env.start_coords[2]
        nvx = 1
        nvy = 1
        reward = -10
    elseif env.track[nx, ny] == 1
        nx = env.start_coords[1]
        ny = env.start_coords[2]
        nvx = 1
        nvy = 1
        reward = -10
    end

    # can't go negative speed
    nvx > 1 || (nvx = 1)
    nvy > 1 || (nvy = 1)

    # can't go too fast
    nvx < 5 || (nvx = 5)
    nvy < 5 || (nvy = 5)

    env.track[nx, ny] != 2 || (reward = 200.0)

    done = reward > 0
    env.state = [nx, ny, nvx, nvy]
    return env.state, reward, done
end

# Cartpole env definition (uses OpenAI gym)
# observation space is discretized in our layer (backend is continious)
mutable struct DiscreteCartpole <: Environment
    action_space::DiscreteSpace
    observation_space::DiscreteSpace
    num_bins::Int64
    theta_dot_min::Float64
    theta_dot_max::Float64
    x_dot_min::Float64
    x_dot_max::Float64
    state::Array{Int32}
end

function DiscreteCartpole(num_bins::Int64)
    py"""
    import gym
    env = gym.make('CartPole-v1')

    def cart_reset():
        return env.reset()

    def cart_step(action):
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        return (observation, reward, done)
    """
    theta_dot_lim = 3.5
    v_dot_lim = 2
    return DiscreteCartpole(DiscreteSpace([2]),
                            DiscreteSpace([num_bins,num_bins,num_bins,num_bins]),
                            num_bins, -theta_dot_lim, theta_dot_lim, -v_dot_lim, v_dot_lim,
                            [1,1,1,1])
end

function state(env::DiscreteCartpole)
    return env.state
end

function action_space(env::DiscreteCartpole)
    return env.action_space
end

function observation_space(env::DiscreteCartpole)
    return env.observation_space
end

# there is no way on earth this is the right way to do this
# but I couldn't find a more effecient way
function discretize(value, min, max, bins)
    range = abs(max - min)
    stride = range / bins

    value < max || return bins
    value > min || return 1

    low_border = min
    high_border = min + stride
    for b in 1:bins
        if value > low_border && value < high_border
            return b
        end
        low_border += stride
        high_border += stride
    end

    return bins
end

function observation_to_state(env::DiscreteCartpole, observation)
    state = [0, 0, 0, 0]
    state[1] = discretize(observation[1], -2.4, 2.4, env.num_bins)
    state[2] = discretize(observation[2], env.x_dot_min, env.x_dot_max, env.num_bins)
    state[3] = discretize(observation[3], -0.218, 0.218, env.num_bins)
    state[4] = discretize(observation[4], env.theta_dot_min, env.theta_dot_max, env.num_bins)
    return state
end

function reset!(env::DiscreteCartpole)
    observation = py"cart_reset"()
    observation = observation[1]
    env.state = observation_to_state(env, observation)
end

function step!(env::DiscreteCartpole, action)
    py_action = action - 1
    observation, reward, done = py"cart_step"(py_action)

    !done || (reward = -100)

    env.state = observation_to_state(env, observation)

    return env.state, reward, done
end


mutable struct Cartpole <: Environment
    action_space::DiscreteSpace
    observation_space::ContinuousSpace
    state::Array{Float64}
end

function Cartpole()
    py"""
    import gym
    env = gym.make('CartPole-v1')

    def cart_reset():
        return env.reset()

    def cart_step(action):
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        return (observation, reward, done)
    """
    theta_dot_lim = 3.5
    v_dot_lim = 2
    return Cartpole(DiscreteSpace([2]),
                    ContinuousSpace([(-2.4, 2.4),(-v_dot_lim, v_dot_lim),(-0.218, 0.218),(-theta_dot_lim, theta_dot_lim)]),
                    [0.0,0.0,0.0,0.0])
end

function state(env::Cartpole)
    return env.state
end

function action_space(env::Cartpole)
    return env.action_space
end

function observation_space(env::Cartpole)
    return env.observation_space
end

function reset!(env::Cartpole)
    observation = py"cart_reset"()
    observation = observation[1]
    env.state = observation
end

function step!(env::Cartpole, action)
    py_action = action - 1
    observation, reward, done = py"cart_step"(py_action)

    !done || (reward = -100)
    env.state = [clamp(observation[1], -2.4, 2.4), clamp(observation[2], -2, 2), clamp(observation[3], -0.218, 0.218), clamp(observation[4], -3.5, 3.5)]

    return env.state, reward, done
end
