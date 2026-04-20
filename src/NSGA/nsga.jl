using Parameters

include("./population.jl")
include("fast_non_dominated_sorting.jl")
include("crowding_distance.jl")
include("selection.jl")
include("prints.jl")

function run_nsga(cfg::NSGAConfig)
    @unpack pop_size, num_gens, n_features, evaluate, tournament_size, crossover_rate, mutation_rate, log_every, verbose = cfg

    if verbose
        print_run_header(cfg)
    end

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

    # Time-series used for experiment tables and plotting.
    f1_size_by_gen = Int[]
    best_accuracy_by_gen = Float64[]
    min_features_by_gen = Int[]

    initial_front = rank1_front(parents)
    initial_stats = front_stats(initial_front)
    push!(f1_size_by_gen, initial_stats.size)
    push!(best_accuracy_by_gen, initial_stats.best_accuracy)
    push!(min_features_by_gen, initial_stats.min_features)
    if verbose
        print_generation_summary("NSGA-II", 0, initial_front)
    end

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

        current_front = rank1_front(parents)
        current_stats = front_stats(current_front)
        push!(f1_size_by_gen, current_stats.size)
        push!(best_accuracy_by_gen, current_stats.best_accuracy)
        push!(min_features_by_gen, current_stats.min_features)

        if verbose && (gen % log_every == 0 || gen == num_gens)
            print_generation_summary("NSGA-II", gen, current_front)
        end

        gen += 1
    end

    final_front = rank1_front(parents)
    if verbose
        print_final_summary(final_front)
    end

    return (
        final_population = parents,
        pareto_front = final_front,
        representative = representative_solution(final_front),
        history = (
            f1_size_by_gen = f1_size_by_gen,
            best_accuracy_by_gen = best_accuracy_by_gen,
            min_features_by_gen = min_features_by_gen
        )
    )
end
