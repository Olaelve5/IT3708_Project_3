struct PSORunResult
    best_solution::BitVector
    best_fitness::Float64
    best_iteration::Int
end

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

function run_pso(config::PSOConfig)::PSORunResult
    swarm = initialize_swarm(config)
    best_fitness = config.evaluate(swarm.global_best)
    best_iteration = 0

    for iter in 1:config.num_iterations
        if config.verbose && config.log_every > 0 && (iter == 1 || iter % config.log_every == 0)
            println("Iteration: ", iter)
        end
        for particle in swarm.particles
            update_velocity!(particle, swarm, config)
            update_position!(particle)
            update_personal_best!(particle, config)
        end
        update_global_best!(swarm, config)

        current_best_fitness = config.evaluate(swarm.global_best)
        if current_best_fitness > best_fitness
            best_fitness = current_best_fitness
            best_iteration = iter
        end
    end

    return PSORunResult(copy(swarm.global_best), best_fitness, best_iteration)
end

function run_pso_experiment(
    config::PSOConfig,
    num_runs::Int;
    dataset_name::String = "Unknown",
    optimal_fitness::Union{Nothing, Float64} = nothing,
    atol::Float64 = 1e-12,
)
    quiet_config = PSOConfig(
        num_particles = config.num_particles,
        num_features = config.num_features,
        num_iterations = config.num_iterations,
        inertia_weight = config.inertia_weight,
        cognitive_coefficient = config.cognitive_coefficient,
        social_coefficient = config.social_coefficient,
        evaluate = config.evaluate,
        log_every = config.log_every,
        verbose = false
    )

    run_results = PSORunResult[]
    best_fitnesses = Float64[]
    best_iterations = Int[]
    success_count = 0

    println("Running PSO $num_runs times on '$dataset_name'...")

    for _ in 1:num_runs
        result = run_pso(quiet_config)
        push!(run_results, result)
        push!(best_fitnesses, result.best_fitness)
        push!(best_iterations, result.best_iteration)
        if !isnothing(optimal_fitness) && isapprox(result.best_fitness, optimal_fitness; atol = atol, rtol = 0.0)
            success_count += 1
        end
        print(".")
    end
    println()

    return (
        dataset_name = dataset_name,
        num_runs = num_runs,
        runs = run_results,
        best_fitness = (
            mean = mean(best_fitnesses),
            std = std(best_fitnesses),
        ),
        best_iteration = (
            mean = mean(best_iterations),
            std = std(best_iterations),
        ),
        success = (
            count = success_count,
            rate = success_count / num_runs,
            optimal_fitness = optimal_fitness,
        ),
    )
end
