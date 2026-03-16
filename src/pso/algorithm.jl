function initialize_particle(config::PSOConfig)::Particle
    position = bitrand(config.num_features)
    velocity = randn(config.num_features)
    Particle(position, velocity, copy(position))
end

function initialize_swarm(config::PSOConfig)::Swarm
  particles = [initialize_particle(config) for _ in 1:config.num_particles]
  best = argmax(p -> config.evaluate(p.position), particles)
  Swarm(particles, copy(best.position))
end

function update_velocity!(particle::Particle, swarm::Swarm, config::PSOConfig)
    r1 = rand(config.num_features)
    r2 = rand(config.num_features)
    particle.velocity = config.inertia_weight .* particle.velocity .+
                        config.cognitive_coefficient .* r1 .* (particle.personal_best .- particle.position) .+
                        config.social_coefficient .* r2 .* (swarm.global_best .- particle.position)
end

function update_position!(particle::Particle)
    for i in eachindex(particle.position)
        p = 1 / (1 + exp(-particle.velocity[i])) # sigmoid
        particle.position[i] = rand() < p
    end
end

function update_personal_best!(particle::Particle, config::PSOConfig)
    if config.evaluate(particle.position) > config.evaluate(particle.personal_best)
        particle.personal_best .= particle.position
    end
end

function update_global_best!(swarm::Swarm, config::PSOConfig)
    for particle in swarm.particles
        if config.evaluate(particle.personal_best) > config.evaluate(swarm.global_best)
            swarm.global_best .= particle.personal_best
        end
    end
end

function run_pso(config::PSOConfig)::BitVector
    swarm = initialize_swarm(config)

    for iter in 1:config.num_iterations
      println("Iteration: ", iter)
        for particle in swarm.particles
            update_velocity!(particle, swarm, config)
            update_position!(particle)
            update_personal_best!(particle, config)
        end
        update_global_best!(swarm, config)
    end

    return swarm.global_best
end





function run_pso
    
end