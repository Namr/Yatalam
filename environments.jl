using DelimitedFiles
using Distributions
using StatsBase

# An implementation of Environment REQUIRES: state, observation_space, action_space
abstract type Environment end
abstract type Space end

struct DiscreteSpace
    size::Array{Int32}
end

struct ContinuousSpace
    size::Array{Tuple{Float32, Float32}}
end

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
