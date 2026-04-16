function make_evaluate(fitness_landscape::Vector{Float64})
    
    function evaluate(position::BitVector)::Float64
        index = 0
        for i in eachindex(position)
            if position[i]
                index += (1 << (i - 1))
            end
        end
        
        if index == 0
            return 0.0
        end
        
        return fitness_landscape[index]
    end

    return evaluate
end