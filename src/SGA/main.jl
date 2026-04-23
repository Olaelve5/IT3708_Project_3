using EvoLP
using Parameters
using Statistics # Required for mean and std
include("../common/population.jl")
include("../common/parse_data.jl")
include("../common/evaluation.jl")
include("../common/config.jl")
include("../common/generate_synthetic.jl")
include("../plotting/calculate_optimas.jl")


function SGA(config::SGAConfig, eval_func::Function, num_features::Int)
    @unpack pop_size, num_gens, tournament_size, mutation_rate, crossover_rate, elitism = config

    S = TournamentSelector(tournament_size)
    M = BitwiseMutator(mutation_rate)
    C = UniformRecombinator()

    population = generate_population(pop_size, num_features)

    # Variables to track the overall best across all generations
    overall_best_fitness = -Inf
    best_iteration = 1
    best_solution = BitVector()

    for gen in 1:num_gens
        fitnesses = [eval_func(ind) for ind in population]

        best_idx = argmax(fitnesses)
        best_gen_fitness = fitnesses[best_idx]
        champion = copy(population[best_idx])
        
        # Update overall best if the current generation's champion is better
        if best_gen_fitness > overall_best_fitness
            overall_best_fitness = best_gen_fitness
            best_iteration = gen
            best_solution = champion
        end

        # avg_fitness = mean(fitnesses)
        # println("Generation $gen | Best Fitness: $best_gen_fitness | Average Fitness: $avg_fitness")
        
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
    
    return best_solution, overall_best_fitness, best_iteration 
end


config = SGAConfig(
    pop_size = 100,
    num_gens = 100,
    tournament_size = 3,
    mutation_rate = 0.01,
    crossover_rate = 0.9,
    elitism = true
)

#acc_vec, landscape, N_FEATURES = parse_file("train_data/01-breast-w_lr_F.h5")
#acc_vec, landscape, N_FEATURES = parse_file("train_data/05-credit-a_rf_F.h5")
#acc_vec, landscape, N_FEATURES = parse_file("train_data/08-letter-r_knn_F.h5")
#acc_vec, landscape, N_FEATURES = parse_file("test_data/06-zoo_lr_F.h5")
#acc_vec, landscape, N_FEATURES = parse_file("test_data/10-hepatitis_lr_F.h5", epsilon=0)
#acc_vec, landscape, N_FEATURES = generate_synthetic_fitness_landscape()

eval_func = make_evaluate(landscape)

# Calculate global optimum for reference
global_opt_fitness, global_opt_index, local_optima_indices = calculate_optimas(landscape, Int(N_FEATURES))
println("Global Optimum Fitness: ", global_opt_fitness)
println("--------------------------------------------------")

# Run the algorithm 10 times
num_runs = 10
best_fitnesses = Float64[]
best_iterations = Int[]

println("Running SGA $num_runs times...")

for r in 1:num_runs
    _, best_fitness, best_iter = SGA(config, eval_func, Int(N_FEATURES))
    push!(best_fitnesses, best_fitness)
    push!(best_iterations, best_iter)
    println("Run $r completed: Best Fitness = $best_fitness (found at generation $best_iter)")
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