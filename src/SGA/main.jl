using EvoLP
using Parameters
using Statistics # Required for mean and std
include("../common/population.jl")
include("../common/parse_data.jl")
include("../common/evaluation.jl")
include("../common/config.jl")
include("../common/generate_synthetic.jl")
include("../plotting/calculate_optimas.jl")
include("../plotting/fitness_plot.jl")


function SGA(config::SGAConfig, eval_func::Function, num_features::Int; verbose::Bool=false)
    @unpack pop_size, num_gens, tournament_size, mutation_rate, crossover_rate, elitism = config

    S = TournamentSelector(tournament_size)
    M = BitwiseMutator(mutation_rate)
    C = UniformRecombinator()

    population = generate_population(pop_size, num_features)

    # Variables to track the overall best across all generations
    overall_best_fitness = -Inf
    best_iteration = 1
    best_solution = BitVector()

    # Arrays to track history for plotting
    history_best = Float64[]
    history_avg = Float64[]

    for gen in 1:num_gens
        fitnesses = [eval_func(ind) for ind in population]

        best_idx = argmax(fitnesses)
        best_gen_fitness = fitnesses[best_idx]
        champion = copy(population[best_idx])
        avg_fitness = mean(fitnesses)
        
        # Record history for this generation
        push!(history_best, best_gen_fitness)
        push!(history_avg, avg_fitness)
        
        # Update overall best if the current generation's champion is better
        if best_gen_fitness > overall_best_fitness
            overall_best_fitness = best_gen_fitness
            best_iteration = gen
            best_solution = champion
        end

        # Print per-generation statistics if verbose mode is on
        if verbose
            println("Generation $gen | Best Fitness: $best_gen_fitness | Average Fitness: $avg_fitness")
        end
        
        new_population = elitism ? BitVector[champion] : BitVector[]
        num_children = elitism ? pop_size - 1 : pop_size
        
        for _ in 1:num_children
            parent_indices = select(S, .-fitnesses)

            if rand() < crossover_rate
                child = cross(C, population[parent_indices[1]], population[parent_indices[2]])
            else
                child = copy(population[parent_indices[1]])
            end
            
            child = mutate(M, child)
            push!(new_population, child)
        end
        
        population = new_population
    end
    
    return best_solution, overall_best_fitness, best_iteration, history_best, history_avg 
end


config = SGAConfig(
    pop_size = 200,
    num_gens = 100,
    tournament_size = 3,
    mutation_rate = 0.01,
    crossover_rate = 0.9,
    elitism = true
)

#acc_vec, landscape, N_FEATURES = parse_file("train_data/01-breast-w_lr_F.h5")
#acc_vec, landscape, N_FEATURES = parse_file("train_data/05-credit-a_rf_F.h5")
#acc_vec, landscape, N_FEATURES = parse_file("train_data/08-letter-r_knn_F.h5")
#acc_vec, landscape, N_FEATURES = generate_train_synthetic()
#acc_vec, landscape, N_FEATURES = parse_file("test_data/06-zoo_lr_F.h5", epsilon=0)
#acc_vec, landscape, N_FEATURES = parse_file("test_data/10-hepatitis_lr_F.h5", epsilon=0)
acc_vec, landscape, N_FEATURES = generate_test_synthetic()

eval_func = make_evaluate(landscape)

# ==========================================
# RUN CONFIGURATION
# ==========================================
# Change this variable to switch modes. 
# Options: :single (1 run, detailed) or :multiple (10 runs, summary)
run_mode = :single 

if run_mode == :single
    println("Running SGA 1 time with per-generation statistics...")
    
    # Unpack the new history arrays returned by SGA
    best_sol, best_fitness, best_iter, hist_best, hist_avg = SGA(config, eval_func, Int(N_FEATURES), verbose=true)
    
    println("\n=== Single Run Complete ===")
    println("Overall Best Fitness: ", best_fitness)
    println("Found at Generation:  ", best_iter)
    println("Best Genome:          ", best_sol)

    # Call the plotting function from your fitness_plot.jl file
    println("\nGenerating fitness plot...")
    plot_fitness_history(hist_avg, hist_best)

    println("Press Enter in the terminal to close the plot and exit.")
    readline()

elseif run_mode == :multiple
    num_runs = 10
    best_fitnesses = Float64[]
    best_iterations = Int[]
    
    # Track the absolute best genome found across all runs
    absolute_best_fitness = -Inf
    absolute_best_genome = BitVector()
    
    println("Running SGA $num_runs times...")
    
    for r in 1:num_runs
        # Set verbose=false and ignore the history arrays returned using `_`
        best_sol, best_fitness, best_iter, _, _ = SGA(config, eval_func, Int(N_FEATURES), verbose=false)
        push!(best_fitnesses, best_fitness)
        push!(best_iterations, best_iter)
        println("Run $r completed: Best Fitness = $best_fitness (found at generation $best_iter)")
        
        # Update the absolute best if this run found a better solution
        if best_fitness > absolute_best_fitness
            global absolute_best_fitness = best_fitness
            global absolute_best_genome = best_sol
        end
    end
    
    # Calculate Statistics
    mean_fitness = mean(best_fitnesses)
    std_fitness = std(best_fitnesses)
    mean_iteration = mean(best_iterations)
    std_iteration = std(best_iterations)
    
    # Print Final Output
    println("\n=== Results over $num_runs runs ===")
    println("Best fitness mean:   ", mean_fitness)
    println("Best fitness Std:    ", std_fitness)
    println("Best iteration mean: ", mean_iteration)
    println("Best iteration std:  ", std_iteration)
    
    println("\n=== Absolute Best Solution Found ===")
    println("Best Fitness: ", absolute_best_fitness)
    println("Best Genome:  ", absolute_best_genome)

else
    println("Invalid run_mode. Please set to :single or :multiple.")
end