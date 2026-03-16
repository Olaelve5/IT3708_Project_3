function initialize_particle(config::PSOConfig)::Particle
    position = bitrand(config.num_features)
    velocity = randn(config.num_features)
    Particle(position, velocity, copy(position))
end

function initialize_swarm(config::PSOConfig)::Swarm
  particles = [initialize_particle(config) for _ in 1:config.num_particles]
  Swarm(particles, copy(particles[1].position))
end





function run_pso
    
end