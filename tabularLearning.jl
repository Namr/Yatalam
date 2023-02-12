function montecarlo_control(env::Environment, iters=100, gamma=1.0, ep=0.3)
    values = []
    policy = randompolicy(track, ep)

    Q = ones(Float64, size(track)[1], size(track)[2], speed_limit+1, speed_limit+1, length(action_space)) * typemin(Float64)
    N = zeros(Float64, size(track)[1], size(track)[2], speed_limit+1, speed_limit+1, length(action_space))

    for i in 1:iters
        episode, step = generate_episode(track, start_coords, policy)

        # record the first visit to a state action pair
        seen = zeros(Int32, size(track)[1], size(track)[2], speed_limit+1, speed_limit+1, length(action_space))
        for t in 1:step
            state, action, reward = episode[t]
            if seen[state[1], state[2], state[3]+1, state[4]+1, action] == 0
                seen[state[1], state[2], state[3]+1, state[4]+1, action] = t
            end
        end

        value = 0
        for t in reverse(1:step)
            state, action, reward = episode[t]
            value = (value * gamma) + reward
            # is this the first visit?
            if seen[state[1], state[2], state[3]+1, state[4]+1, action] == t
                N[state[1], state[2], state[3]+1, state[4]+1, action] += 1.0
                # update Q towards the seen value of that action state pair
                Q[state[1], state[2], state[3]+1, state[4]+1, action] += (value - Q[state[1], state[2], state[3]+1, state[4]+1, action]) / N[state[1], state[2], state[3]+1, state[4]+1, action]

                # update policy to be max of Q
                val, oa = findmax(Q[state[1], state[2], state[3]+1, state[4]+1, 1:length(action_space)])

                for a in 1:length(action_space)
                    if a == oa
                        policy[state[1], state[2], state[3]+1, state[4]+1][a] = 1 - ep + (ep/length(action_space))
                    else
                        policy[state[1], state[2], state[3]+1, state[4]+1][a] = ep/length(action_space)
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
