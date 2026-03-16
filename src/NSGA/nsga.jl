using Parameters

include("./population.jl")
include("non_dominated_sorting.jl")
include("crowding_distance.jl")

function run_nsga(cfg::NSGAConfig)
    @unpack pop_size, n_features, evaluate = cfg

    population = init_pop(pop_size, n_features, evaluate)

    while !finished
        # Perform non-dominated sorting
        nd_sort!(population)
        # TODO: Compute crowding distance
        crowding_distance!(population)
        # TODO: Select parents (tournament)
        # TODO: Crossover
        # TODO: Mutation
        # TODO: Evaluate offspring
        # TODO: Combine parent + offspring populations
        # TODO: Sort again
        # TODO: Select next generation
    end
end