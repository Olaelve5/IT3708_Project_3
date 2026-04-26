function calculate_optimas(landscape, n_features::Int)
    local_optima_indices = Int[]
    n_combinations = length(landscape)

    for i in 1:n_combinations
        is_local_optimum = true
        
        # Check all combinations that are exactly 1 bit-flip away
        for bit in 0:(n_features-1)
            neighbor_idx = i ⊻ (1 << bit)
            
            if neighbor_idx > 0 && neighbor_idx <= n_combinations
                if landscape[neighbor_idx] > landscape[i]
                    is_local_optimum = false
                    break
                end
            end
        end
        
        if is_local_optimum
            push!(local_optima_indices, i)
        end
    end
    
    global_opt_fitness, global_opt_index = findmax(landscape)
    
    return global_opt_fitness, global_opt_index, local_optima_indices
end