using EvoLP

# Parameters
POP_SIZE = 1000
TOURNAMENT_SIZE = 3

# Bitvector population generator
function generate_population(pop_size::Int, chromosome_length::Int)::Vector{BitVector}
    return binary_vector_pop(pop_size, chromosome_length)
end