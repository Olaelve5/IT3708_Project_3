using EvoLP
using Parameters
include("../common/population.jl")
include("../common/parse_data.jl")
include("../common/evaluation.jl")
include("../common/config.jl")


function SGA(config::SGAConfig, eval_func::Function, num_features::Int)
    @unpack pop_size, num_gens, tournament_size, mutation_rate, crossover_rate, elitism = config

    S = TournamentSelector(tournament_size)
    M = BitwiseMutator(mutation_rate)
    C = UniformRecombinator()

    population = generate_population(pop_size, num_features)

    for gen in 1:num_gens
        fitnesses = [eval_func(ind) for ind in population]

        best_idx = argmax(fitnesses)
        best_gen_fitness = fitnesses[best_idx]
        champion = copy(population[best_idx])
        avg_fitness = mean(fitnesses)
        println("Generation $gen | Best Fitness: $best_gen_fitness | Average Fitness: $avg_fitness")
        
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
    
    final_fitnesses = [eval_func(ind) for ind in population]
    best_idx = argmax(final_fitnesses)
    
    println("--- Evolution Complete ---")
    println("Best Fitness: ", final_fitnesses[best_idx])
    
    return population[best_idx], final_fitnesses[best_idx] 
end


config = SGAConfig(
    pop_size = 100,
    num_gens = 1000,
    tournament_size = 3,
    mutation_rate = 0.01,
    crossover_rate = 0.9,
    elitism = false
)

#landscape = parse_file("train_data/01-breast-w_lr_F.h5")
#landscape, _ = parse_file("train_data/05-credit-a_rf_F.h5")
#acc_vec, landscape, N_FEATURES = parse_file("train_data/08-letter-r_knn_F.h5")
#acc_vec, landscape, N_FEATURES = parse_file("train_data/05-credit-a_rf_F.h5")
acc_vec, landscape, N_FEATURES = parse_file("train_data/01-breast-w_lr_F.h5")
eval_func = make_evaluate(landscape)

best_solution, best_fitness = SGA(config, eval_func, Int(N_FEATURES))