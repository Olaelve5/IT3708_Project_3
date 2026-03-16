mutable struct Individual
    genome::Bitvector
    accuracy::Float64
    num_features::Int
    rank::Int
    crowding_distance::Float64
end

function new_individual(chromosome::Bitvector, accuracy::Float64)::Individual
    return Individual(chromosome, accuracy, sum(chromosome), 0, 0.0)
end

function init_pop(pop_size::Int, n_features::Int, evaluate::Function)::Vector{Individual}
    population = Vector{Individual}(undef, pop_size)
    genes = generate_population(pop_size, n_features)
    for i in 1:pop_size
        population[i] = new_individual(genes[i], evaluate(genes))
    end
end