using Distributions
using StatsBase
using LinearAlgebra

# Tile coding implementation adapted from Dr. Sutton's implementation on his website
function tile_coding_feature_selector(tiles_per_dim, value_limits, tilings)
  offset(n) = 2 * collect(0:n-1) .+ 1
  tiling_dims = ceil.(tiles_per_dim) .+ 1

  # Spain with no S
  offsets = (offset(length(tiles_per_dim)) .* repeat(collect(0:tilings-1), outer=(1,length(tiles_per_dim)))' )' ./ Float64(tilings) .% 1
  norm_dims = tiles_per_dim ./ (value_limits[2, :] - value_limits[1, :])
  tile_base_ind = prod(tiling_dims) * collect(0:tilings-1)
  hash_vec = [prod(tiling_dims[1:i]) for i in 0:length(tiles_per_dim)-1]
  n_tiles = tilings * prod(tiling_dims)
  println(n_tiles)
  feature_selector(state, action) = begin
    x = [state; action]
    off_coords = trunc.(Int32, ((x .- value_limits[1,:]) .* norm_dims .+ offsets')')
    binary = tile_base_ind .+ (off_coords * hash_vec)
    s = zeros(Float64, n_tiles)
    for b in binary
      s[b] = 1
    end
    return s
  end

  return feature_selector
end

function rbf_feature_selector(input_min, input_max, gamma_min, gamma_max, feature_count)
    #widths = collect(range(gamma_min, gamma_max, length=feature_count))
    widths = [rand(Uniform(gamma_min, gamma_max)) for i in 1:feature_count]
    centers = []
    for f in 1:feature_count
        center = [rand(Uniform(input_min[i], input_max[i])) for i in 1:length(input_min)]
        push!(centers, center)
    end

    feature_selector(state, action) = begin
        input = [state; action]
        ss = []
        for i in 1:length(widths)
            width = widths[i]
            center = centers[i]
            s = exp(-(norm(input - center)^2)/(2*(width^2)))
            push!(ss, s)
        end
        return ss
    end

    return feature_selector
end

function linear_montecarlo_control(env::Environment, feature_selector, alpha=0.2, iters=100, ep=0.3)
    values = []

    reset!(env)
    actions = action_space(env).size[1]
    params = length(feature_selector(state(env), rand(1:actions)))
    w = ones(Float64, params)

    for i in 1:iters
        reset!(env)
        history = Array{Tuple{typeof(state(env)), Int32, Float64}}(undef, 500000)
        step = 1
        while true
            # find best action in this state
            # TODO: make this work for more than one action
            maxv = -1000000
            maxa = 1
            action_weight = [ep/actions for a in 1:actions]
            Q = [transpose(w) * feature_selector(state(env), a) for a in 1:actions]
            oa = argmax(Q)
            action_weight[oa] = 1.0 - ep + (ep/actions)
            action = StatsBase.sample(1:actions, Weights(action_weight))

            s, reward, done = step!(env, action)
            history[step] = (s, action, reward)
            !done || break

            step += 1
            if step > 500000
                reset!(env)
                step = 1
            end
        end

        value = 0
        for t in reverse(1:step)
            s, action, reward = history[t]
            value = value + reward

            # update Q towards the seen value of the state-action pair (see Sutton page 205)
            w += alpha * (value - (transpose(w) * feature_selector(s, action))) * feature_selector(s, action)
        end

        append!(values, value)

        print("*")
        flush(stdout)
    end # end iters

    return values
end

function linear_sarsa_control(env::Environment, feature_selector, alpha=0.2, steps=1000, ep=0.3)
    values = []

    reset!(env)
    actions = action_space(env).size[1]
    params = length(feature_selector(state(env), rand(1:actions)))
    w = ones(Float64, params)

    use_policy(s) = begin
        action_weight = [ep/actions for a in 1:actions]
        Q = [transpose(w) * feature_selector(s, a) for a in 1:actions]
        oa = argmax(Q)
        action_weight[oa] = 1.0 - ep + (ep/actions)
        action = StatsBase.sample(1:actions, Weights(action_weight))
        return action
    end

    step = 0
    value = 0
    reset!(env)

    # we assume an action has already been taken before each loop of SARSA
    action = use_policy(state(env))
    s = copy(state(env))
    while true
        # take another action and call the new state s_prime
        s_prime, reward, done = step!(env, action)
        value += reward
        a_prime = use_policy(s_prime)

        # SARSA update
        xt = feature_selector(s, action)
        xt1 = feature_selector(s_prime, a_prime)
        w += alpha * (reward + transpose(w)*xt1 - transpose(w)*xt)*xt

        # we already know what action and state are next to we can set them here
        s = s_prime
        action = a_prime

        step += 1

        if done
            append!(values, value)

            value = 0
            reset!(env)
            action = use_policy(state(env))
            s = copy(state(env))
        end

        step < steps || break
    end

    return values
end


function linear_Qlearning(env::Environment, feature_selector, alpha=0.2, steps=1000, ep=0.3)
    values = []

    reset!(env)
    actions = action_space(env).size[1]
    params = length(feature_selector(state(env), rand(1:actions)))
    w = ones(Float64, params)

    use_policy(s) = begin
        action_weight = [ep/actions for a in 1:actions]
        Q = [transpose(w) * feature_selector(s,a) for a in 1:actions]
        oa = argmax(Q)
        action_weight[oa] = 1.0 - ep + (ep/actions)
        action = StatsBase.sample(1:actions, Weights(action_weight))
        return action
    end

    step = 0
    value = 0
    reset!(env)

    # we assume an action has already been taken before each loop of SARSA
    action = use_policy(state(env))
    s = copy(state(env))
    while true
        # take another action and call the new state s_prime
        s_prime, reward, done = step!(env, action)
        value += reward
        a_prime = use_policy(s_prime)

        # SARSA update
        xt = feature_selector(s, action)

        maxv = -1000000
        for a in 1:actions
            v = transpose(w) * feature_selector(s_prime, a)
            if v > maxv
                maxv = v
            end
        end

        w += alpha * (reward + maxv - transpose(w)*xt)*xt

        # we already know what action and state are next to we can set them here
        s = s_prime
        action = a_prime

        step += 1

        if done
            append!(values, value)

            value = 0
            reset!(env)
            action = use_policy(state(env))
            s = copy(state(env))
            #ep *= 0.9
        end

        step < steps || break
    end

    return values
end
