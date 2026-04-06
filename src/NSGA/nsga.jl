using Parameters

include("./population.jl")
include("fast_non_dominated_sorting.jl")
include("crowding_distance.jl")
include("selection.jl")

function run_nsga(cfg::NSGAConfig)
    @unpack pop_size, n_features, evaluate, tournament_size, crossover_rate, mutation_rate = cfg

    # Define genetic operators
    const SELECTOR = NSGATournament(tournament_size)

    # Generate, sort and evaluate initial parent population
    parents = init_pop(pop_size, n_features, evaluate)
    fnd_sort!(parents, pop_size)
    crowding_distance!(parents)

    # Generate initial offspring population from initial parent population
    offspring = Vector{Individual}(undef, pop_size)
    generate_offspring!(parents, offspring, SELECTOR, crossover_rate, mutation_rate)

    # Initialize reusable vectors for population in algorithm
    population = Vector{Individual}(undef, 2*pop_size)
    new_parents = Vector{Individual}(undef, pop_size)


    finished = false
    while !finished
        # Combine parent + offspring populations
        for i in 1:pop_size
            population[i] = parents[i]
            population[pop_size+i] = offspring[i]
        end
        
        # Perform non-dominated sorting
        fnd_sort!(population, pop_size)

        # Create next parent generation based on rank and crowding
        next_idx = 1
        i = 1
        while length(new_parents) < pop_size
            front_i = [p for p in population if p.rank == i]
            # Compute per front crowding distance
            crowding_distance!(front_i)
            for p in front_i
                next_idx > pop_size && break
                new_parents[next_idx] = p
                next_idx += 1
            end
            i += 1
        end

        generate_offspring!(new_parents, offspring, SELECTOR, crossover_rate, mutation_rate)
        # TODO: Evaluate offspring
        # TODO: Sort again
        # TODO: Select next generation
        if something
            finished = true
        end
    end
end
