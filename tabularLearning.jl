using Distributions
using StatsBase

function tabular_montecarlo_control(env::Environment, iters=100, gamma=1.0, ep=0.3)
    values = []

    # init a policy table [state, action] -> probability of action in state
    states = prod(env.observation_space.size)
    actions = prod(env.action_space.size)
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
        dims = length(env.observation_space.size)
        for d in 1:dims- 1
            i += s[d] * prod(env.observation_space.size[d+1:dims])
        end
        i += s[dims]
        return i
    end
    use_policy(state) = sample(1:actions, Weights(policy[state_to_index(state),:]))

    # tables to keep track of value of state-action pairs
    # and the # of times they have been visited
    Q = ones(Float64, states, actions) * typemin(Float64)
    N = zeros(Float64, states, actions)

    for i in 1:iters
        # generate an episode:
        history = Array{Tuple{typeof(env.state), Int32, Float64}}(undef, 500000)
        step = 1
        while true
            action = use_policy(env.state)
            state, reward, done = step!(env, action)
            history[step] = (state, action, reward)

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
            state, action, reward = history[t]
            if seen[state_to_index(state), action] == 0
                seen[state_to_index(state), action] = t
            end
        end

        value = 0
        for t in reverse(1:step)
            state, action, reward = history[t]
            value = (value * gamma) + reward
            # is this the first visit?
            if seen[state_to_index(state), action] == t
                N[state_to_index(state), action] += 1.0
                # update Q towards the seen value of that action state pair
                Q[state_to_index(state), action] += (value - Q[state_to_index(state), action]) / N[state_to_index(state), action]

                # update policy to be max of Q
                val, oa = findmax(Q[state_to_index(state), :])

                for a in 1:actions
                    if a == oa
                        policy[state_to_index(state), a] = 1 - ep + (ep/actions)
                    else
                        policy[state_to_index(state), a] = ep/actions
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

function tabular_sarsa_control(env::Environment, alpha=0.2, iters=100, gamma=1.0, ep=0.3)
    values = []

    # init a policy table [state, action] -> probability of action in state
    states = prod(env.observation_space.size)
    actions = prod(env.action_space.size)
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
        dims = length(env.observation_space.size)
        for d in 1:dims- 1
            i += s[d] * prod(env.observation_space.size[d+1:dims])
        end
        i += s[dims]
        return i
    end
    use_policy(state) = sample(1:actions, Weights(policy[state_to_index(state),:]))

    # table to keep track of value of state-action pairs
    Q = ones(Float64, states, actions) * typemin(Float64)

    # TODO: there should be no outer loop of SARSA (doesn't need terminated episodes)
    for i in 1:iters
        value = 0

        # we assume an action has already been taken before each loop of SARSA
        action = use_policy(env.state)
        state = copy(env.state)
        while true

            # take another action and call the new state s_prime
            s_prime, reward, done = step!(env, action)
            value += reward
            a_prime = use_policy(s_prime)

            # SARSA update
            Q[state_to_index(state), action] += alpha * (reward +
                (gamma * Q[state_to_index(s_prime), a_prime]) - Q[state_to_index(state), action])

            # update policy according to new value information
            # find best action in original state (pre-SARSA update)
            val, oa = findmax(Q[state_to_index(state), :])

            for a in 1:actions
                if a == oa
                    policy[state_to_index(state), a] = 1 - ep + (ep/actions)
                else
                    policy[state_to_index(state), a] = ep/actions
                end
            end

            # we already know what action and state are next to we can set them here
            state = s_prime
            action = a_prime

            !done || break
        end

        append!(values, value)
        print("*")
        flush(stdout)
    end

    return values
end
