using EvoLP
include("../common/common.jl")
include("../common/parse_data.jl")

landscape = parse_file("train_data/05-credit-a_rf_F.h5")

POP_SIZE = 1000
N_FEATURES = round(Int, log2(length(landscape) + 1))


function SGA(pop_size::Int, tournament_size::Int, eval_func::Function, num_gens::Int, num_features::Int)
    population = generate_population(pop_size, num_features)

    for gen in 1:num_gens
        println("Generation $gen")
        fitnesses = [eval_func(ind) for ind in population]
        
        new_population = BitVector[]
        
        for _ in 1:pop_size
            parent1 = tournament_selection(population, fitnesses, tournament_size)
            parent2 = tournament_selection(population, fitnesses, tournament_size)
            
            child = uniform_crossover(parent1, parent2)
            child = bitflip_mutation(child, 0.01)
            
            push!(new_population, child)
        end
        
        population = new_population
    end
    
    # return the best solution found
    final_fitnesses = [eval_func(ind) for ind in population]
    best_idx = argmax(final_fitnesses)
    println("Best solution found: ", population[best_idx], " with fitness ", final_fitnesses[best_idx])
    
    return population[best_idx], final_fitnesses[best_idx]
    
end


SGA(POP_SIZE, TOURNAMENT_SIZE, ind -> landscape[parse(Int, string(ind), base=2)], 50, N_FEATURES)

