using Parameters

include("./population.jl")
include("fast_non_dominated_sorting.jl")
include("crowding_distance.jl")
include("selection.jl")

function run_nsga(cfg::NSGAConfig)
    @unpack pop_size, num_gens, n_features, evaluate, tournament_size, crossover_rate, mutation_rate = cfg

    # Define genetic operators
    SELECTOR = NSGATournament(tournament_size)
    MUTATOR = EvoLP.BitwiseMutator(mutation_rate)

    # Generate, sort and evaluate initial parent population
    parents = init_pop(pop_size, n_features, evaluate)
    fnd_sort!(parents, pop_size)
    crowding_distance!(parents)

    # Generate initial offspring population from initial parent population
    offspring = Vector{Individual}(undef, pop_size)
    generate_offspring!(parents, offspring, SELECTOR, crossover_rate, MUTATOR, evaluate)

    # Initialize reusable vectors for population in algorithm
    population = Vector{Individual}(undef, 2*pop_size)
    new_parents = Vector{Individual}(undef, pop_size)


    gen = 1
    while gen <= num_gens
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
        while next_idx <= pop_size
            front_i = [p for p in population if p.rank == i]
            isempty(front_i) && break

            # If front doesnt fit, sort and keep best
            if next_idx + length(front_i) - 1 > pop_size
                # Compute per front crowding distance
                front_crowding_distance!(front_i)
                sort!(front_i, by = p -> p.crowding_distance, rev=true)
                remaining = pop_size - next_idx + 1
                for j in 1:remaining
                    new_parents[next_idx] = front_i[j]
                    next_idx += 1
                end
            else
                for p in front_i
                    new_parents[next_idx] = p
                    next_idx += 1
                end
            end
            i += 1
        end

        # Update crowding_distance of new_parents, rank should stay correct
        crowding_distance!(new_parents)
        # Generate offspring for next generation
        generate_offspring!(new_parents, offspring, SELECTOR, crossover_rate, MUTATOR, evaluate)

        parents .= new_parents
        gen += 1
    end
    return parents
end
