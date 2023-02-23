using Distributions
using StatsBase
using Flux: train!

function linear_montecarlo_control(env::Environment, iters=100, gamma=1.0, ep=0.3)
  values = []

  # init a linear function approximator
  # f(state, action) -> value
  states = length(observation_space(env).size)
  #actions = length(action_space(env).size)
  Q = Dense(states + 1, 1, identity)

  loss(model, x, y) = y - model(x)[1]
  opt = Descent()

  for i in 1:iters
    # generate an episode:
    reset!(env)
    history = Array{Tuple{typeof(state(env)), Int32, Float64}}(undef, 500000)
    step = 1
    while true
      maxv = -10000000
      maxa = 1
      for a in 1:action_space(env).size[1]
        println([state(env); a])
        v = Q([state(env); a])[1]
        if v > maxv
          maxv = v
          maxa = a
        end
      end

      s, reward, done = step!(env, maxa)
      history[step] = (s, maxa, reward)

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
        value = (value * gamma) + reward
        
        # update Q towards the seen value of that action state pair
        train!(loss, Q, [([s; action], value)], opt)
    end

    append!(values, value)

    #policy.epsilon = policy.epsilon / 1.05

    print("*")
    flush(stdout)
  end

  return values
end
