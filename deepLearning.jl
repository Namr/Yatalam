using Flux
using StatsBase
using Distributions


function deep_Qlearning(env::Environment, steps=5000, batch_size=64, ep=0.3)
    values = []
    reset!(env)
    actions = length(action_space(env).size[1])
    states = length(observation_space(env).size)
    inputs = actions + states

    Q = Chain(
            Dense(inputs => 64, sigmoid),
            Dense(64 => 128, relu),
            Dense(128 => 128, relu),
            Dense(128 => 1, relu)
             )

    optim = Flux.setup(Flux.Adam(0.01), Q)

    max_q(s) = begin
        maxv = -1000000
        maxa = 1
        for a in 1:actions
            v = Q([s; a])[1]
            if v > maxv
                maxv = v
                maxa = a
            end
        end

        return maxv, maxa
    end

    use_policy(s) = begin
        maxv, maxa = max_q(s)
        action_weight = [ep/actions for a in 1:actions]
        action_weight[maxa] = 1.0 - ep + (ep/actions)
        action = StatsBase.sample(1:actions, Weights(action_weight))
        return action
    end

    step = 0
    value = 0

    # we assume an action has already been taken before each loop of SARSA
    action = use_policy(state(env))
    s = copy(state(env))
    memory = []
    while true
        # take another action and call the new state s_prime
        s_prime, reward, done = step!(env, action)

        value += reward
        a_prime = use_policy(s_prime)
        push!(memory, (s, action, s_prime, reward, done))

        step += 1

        if done
            # sample from memory buffer
            if length(memory) > batch_size
                batch = StatsBase.sample(memory, batch_size)
                for (state, action, s_prime, reward, done) in batch
                    # train off of one step
                    next_q, next_a = max_q(s_prime)
                    q_value = reward + next_q

                    grads = Flux.gradient(Q) do m
                        result = m([s; action])
                        Flux.mse(result, q_value)
                    end
                    Flux.update!(optim, Q, grads[1])
                end
            end

            append!(values, value)

            print("*")
            flush(stdout)

            value = 0
            reset!(env)
            action = use_policy(state(env))
            s = copy(state(env))
        else
            # train off of one step
            next_q, next_a = max_q(s_prime)
            q_value = reward + next_q

            grads = Flux.gradient(Q) do m
                result = m([s; action])
                Flux.mse(result, q_value)
            end
            Flux.update!(optim, Q, grads[1])
        end

        step < steps || break
    end

    return values
end
