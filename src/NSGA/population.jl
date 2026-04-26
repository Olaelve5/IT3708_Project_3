mutable struct Individual
    genome::BitVector
    accuracy::Real
    num_features::Int
    rank::Int # Assigned rank
    crowding_distance::Float64
    S_p::Vector{Individual} # Dominated solutions
    n_p::Int # n individuals dominating self
end

function ensure_nonempty_genome!(genome::BitVector)
    if count(genome) == 0
        genome[rand(1:length(genome))] = true
    end
    return genome
end

function new_individual(chromosome::BitVector, accuracy::Real)::Individual
    return Individual(chromosome, accuracy, count(chromosome), 0, 0.0, [], 0)
end

function init_pop(
    pop_size::Int,
    n_features::Int,
    evaluate::Function)::Vector{Individual}

    population = Vector{Individual}(undef, pop_size)
    pop_genes = generate_population(pop_size, n_features)
    for i in 1:pop_size
        ensure_nonempty_genome!(pop_genes[i])
        population[i] = new_individual(pop_genes[i], evaluate(pop_genes[i]))
    end
    return population
end