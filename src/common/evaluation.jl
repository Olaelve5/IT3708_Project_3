function make_evaluate(fitness_landscape::Vector{Float64})
    function evaluate(position::BitVector)::Float64
        index = sum(position[i] * 2^(i-1) for i in eachindex(position)) + 1
        return fitness_landscape[index]
    end

    return evaluate
end