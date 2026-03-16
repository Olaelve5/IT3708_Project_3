function calculate_optimas(landscape)
    optima_indices = Int[]

    for i in 1:n_combinations
        is_local_optimum = true
        
        # Check all combinations that are exactly 1 bit-flip away
        for bit in 0:(n_features-1)
            # Flip one bit using XOR
            neighbor_idx = i ⊻ (1 << bit)
            
            # Ensure the neighbor is within our array bounds
            if neighbor_idx > 0 && neighbor_idx <= n_combinations
                if landscape[neighbor_idx] > landscape[i]
                    is_local_optimum = false
                    break
                end
            end
        end
        
        if is_local_optimum
            push!(optima_indices, i)
        end
    end
    return optima_indices
end