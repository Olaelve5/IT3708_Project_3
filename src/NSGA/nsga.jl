include("./common.jl")

struct Individual
    genome::Bitvector
    accuracy::Float64
    num_features::Int
    rank::Int
    crowding_distance::Float64
end

function new_individual(chromosome::Bitvector)::Individual
    return Individual(chromosome, 0.0, sum(chromosome), 0, 0.0)
end

function init_pop(pop_size::Int)::Vector{Individual}
    population = Vector{Individual}(undef, pop_size)
    genes = generate_population(pop_size, 16)
    for i in 1:pop_size
        population[i] = new_individual(genes[i])
    end
end


population = generate_population(POP_SIZE, 9)
# TODO: Call load function ??

while !finished
    # TODO: Perform non-dominated sorting
    # TODO: Compute crowding distance
    # TODO: Select parents (tournament)
    # TODO: Crossover
    # TODO: Mutation
    # TODO: Evaluate offspring
    # TODO: Combine parent + offspring populations
    # TODO: Sort again
    # TODO: Select next generation
end