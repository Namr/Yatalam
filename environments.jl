using Distributions
using StatsBase

abstract type Environment end
abstract type Space end

struct DiscreteSpace
    size::Array{Int32}
end

struct ContinuousSpace
    size::Array{Tuple{Float32, Float32}}
end

struct Racetrack
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
    open(filepath) do rt
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
    return RaceTrack(track, width, height,
                     DiscreteSpace([7]), DiscreteSpace([width, height, 6, 6]),
                     [start_coords[1], start_coords[2], 1, 1], start_coords, action_mapping)
end

function step!(env::Racetrack, action)
    x, y, vx, vy = env.state
    nx = x + vx
    ny = y + vy
    nvx = vx + env.action_mapping[action][1]
    nvy = vy + env.action_mapping[action][2]
    reward = -1

    # can't go off the track
    if nx > env.width || ny > env.height || nx < 1 || ny < 1
        nx = env.start_coords[1]
        ny = env.start_coords[2]
        nvx = 0
        nvy = 0
        reward = -10
    elseif racetrack[nx, ny] == 1
        nx = env.start_coords[1]
        ny = env.start_coords[2]
        nvx = 0
        nvy = 0
        reward = -10
    end

    # can't go negative speed
    nvx > 0 || (nvx = 0)
    nvy > 0 || (nvy = 0)

    # can't go too fast
    nvx < 5 || (nvx = 5)
    nvy < 5 || (nvy = 5)

    racetrack[nx, ny] != 2 || (reward = 200.0)

    return (nx, ny, nvx, nvy), reward
end
