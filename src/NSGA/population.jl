struct Individual
    genome::BitVector
    accuracy::Float64
    num_features::Int
    rank::Int
    crowding_distance::Float64
end

function new_individual(chromosome::BitVector)::Individual
    return Individual(chromosome, 0.0, sum(chromosome), 0, 0.0)
end

function init_pop(pop_size::Int)::Vector{Individual}
    population = Vector{Individual}(undef, pop_size)
    genes = generate_population(pop_size, 16)
    for i in 1:pop_size
        population[i] = new_individual(genes[i])
    end
end