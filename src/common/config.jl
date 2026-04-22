using Parameters

if !isdefined(@__MODULE__, :SGAConfig)
    @with_kw struct SGAConfig
        pop_size::Int = 1000
        num_gens::Int = 50
        tournament_size::Int = 3
        mutation_rate::Float64 = 0.01
        crossover_rate::Float64 = 0.8
        elitism::Bool = true
    end
end

if !isdefined(@__MODULE__, :NSGAConfig)
    @with_kw struct NSGAConfig
        pop_size::Int = 50
        num_gens::Int = 1000
        tournament_size::Int = 3
        n_features::Int
        evaluate::Function
        crossover_rate::Float64 = 0.9
        mutation_rate::Float64
        log_every::Int = 10
        verbose::Bool = true
    end
end

if !isdefined(@__MODULE__, :PSOConfig)
    @with_kw struct PSOConfig
        num_particles::Int = 30
        num_features::Int
        num_iterations::Int = 30
        inertia_weight::Float64 = 0.7
        cognitive_coefficient::Float64 = 1.5
        social_coefficient::Float64 = 1.5
        evaluate::Function
        log_every::Int = 1
        verbose::Bool = true
    end
end
