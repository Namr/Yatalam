using Distributions
using StatsBase

function tabular_montecarlo_control(env::Environment, iters=100, gamma=1.0, ep=0.3)
    values = []

    # init a policy table [state, action] -> probability of action in state
    states = prod(observation_space(env).size)
    actions = prod(action_space(env).size)
    policy = Array{Float64}(undef, states, actions)
    for s in 1:states
        chosen = rand(1:actions)
        for a in 1:actions
            if a == chosen
                policy[s, a] = 1.0 - ep + (ep/actions)
            else
                policy[s, a] = ep/actions
            end
        end
    end

    # helper clojures
    state_to_index(s) = begin
        i = 1
        dims = length(observation_space(env).size)
        for d in 1:dims-1
            i += (s[d]-1) * prod(observation_space(env).size[d+1:dims])
        end
        i += s[dims]
        return i
    end

    use_policy(s) = StatsBase.sample(1:actions, Weights(policy[state_to_index(s),:]))

    # tables to keep track of value of state-action pairs
    # and the # of times they have been visited
    Q = ones(Float64, states, actions) * typemin(Float64)
    N = zeros(Float64, states, actions)

    for i in 1:iters
        # generate an episode:
        reset!(env)
        history = Array{Tuple{typeof(state(env)), Int32, Float64}}(undef, 500000)
        step = 1
        while true
            action = use_policy(state(env))
            s, reward, done = step!(env, action)
            history[step] = (s, action, reward)

            !done || break

            step += 1
            if step > 500000
                reset!(env)
                step = 1
            end
        end

        # record the first visit to a state action pair
        seen = zeros(Int32, states, actions)
        for t in 1:step
            s, action, reward = history[t]
            if seen[state_to_index(s), action] == 0
                seen[state_to_index(s), action] = t
            end
        end

        value = 0
        for t in reverse(1:step)
            s, action, reward = history[t]
            value = (value * gamma) + reward
            # is this the first visit?
            if seen[state_to_index(s), action] == t
                N[state_to_index(s), action] += 1.0 # update Q towards the seen value of that action state pair
                Q[state_to_index(s), action] += (value - Q[state_to_index(s), action]) / N[state_to_index(s), action]

                # update policy to be max of Q
                val, oa = findmax(Q[state_to_index(s), :])

                for a in 1:actions
                    if a == oa
                        policy[state_to_index(s), a] = 1 - ep + (ep/actions)
                    else
                        policy[state_to_index(s), a] = ep/actions
                    end
                end

            end
        end

        append!(values, value)

        #policy.epsilon = policy.epsilon / 1.05

        print("*")
        flush(stdout)
    end

    return values
end

function tabular_sarsa_control(env::Environment, alpha=0.2, steps=1000, gamma=1.0, ep=0.3)
    values = []

    # init a policy table [state, action] -> probability of action in state
    states = prod(observation_space(env).size)
    actions = prod(action_space(env).size)
    policy = Array{Float64}(undef, states, actions)
    for s in 1:states
        chosen = rand(1:actions)
        for a in 1:actions
            if a == chosen
                policy[s, a] = 1.0 - ep + (ep/actions)
            else
                policy[s, a] = ep/actions
            end
        end
    end

    # helper clojures
    state_to_index(s) = begin
        i = 1
        dims = length(observation_space(env).size)
        for d in 1:dims-1
            i += (s[d]-1) * prod(observation_space(env).size[d+1:dims])
        end
        i += s[dims]
        return i
    end

    use_policy(s) = StatsBase.sample(1:actions, Weights(policy[state_to_index(s),:]))

    # table to keep track of value of state-action pairs
    Q = zeros(Float64, states, actions)

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
        Q[state_to_index(s), action] += alpha * (reward +
            (gamma * Q[state_to_index(s_prime), a_prime]) - Q[state_to_index(s), action])

        # update policy according to new value information
        # find best action in original state (pre-SARSA update)
        val, oa = findmax(Q[state_to_index(s), :])

        for a in 1:actions
            if a == oa
                policy[state_to_index(s), a] = 1 - ep + (ep/actions)
            else
                policy[state_to_index(s), a] = ep/actions
            end
        end

        # we already know what action and state are next to we can set them here
        s = s_prime
        action = a_prime

        step += 1

        if done
            append!(values, value)
            print("*")
            flush(stdout)

            value = 0
            reset!(env)
            action = use_policy(state(env))
            s = copy(state(env))
        end

        step < steps || break
    end

    return values
end


function tabular_Qlearning(env::Environment, alpha=0.2, steps=1000, gamma=1.0, ep=0.3)
    values = []

    # init a policy table [state, action] -> probability of action in state
    states = prod(observation_space(env).size)
    actions = prod(action_space(env).size)
    policy = Array{Float64}(undef, states, actions)
    for s in 1:states
        chosen = rand(1:actions)
        for a in 1:actions
            if a == chosen
                policy[s, a] = 1.0 - ep + (ep/actions)
            else
                policy[s, a] = ep/actions
            end
        end
    end

    # helper clojures
    state_to_index(s) = begin
        i = 1
        dims = length(observation_space(env).size)
        for d in 1:dims-1
            i += (s[d]-1) * prod(observation_space(env).size[d+1:dims])
        end
        i += s[dims]
        return i
    end

    use_policy(s) = StatsBase.sample(1:actions, Weights(policy[state_to_index(s),:]))

    # table to keep track of value of state-action pairs
    Q = zeros(Float64, states, actions)

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
        maxQ, oa = findmax(Q[state_to_index(s_prime), :])
        Q[state_to_index(s), action] += alpha * (reward +
                        (gamma * maxQ - Q[state_to_index(s), action]))

        # update policy according to new value information
        # find best action in original state (pre-SARSA update)
        val, oa = findmax(Q[state_to_index(s), :])

        for a in 1:actions
            if a == oa
                policy[state_to_index(s), a] = 1 - ep + (ep/actions)
            else
                policy[state_to_index(s), a] = ep/actions
            end
        end

        # we already know what action and state are next to we can set them here
        s = s_prime
        action = a_prime

        step += 1

        if done
            append!(values, value)
            print("*")
            flush(stdout)

            value = 0
            reset!(env)
            action = use_policy(state(env))
            s = copy(state(env))
        end

        step < steps || break
    end

    return values
end
