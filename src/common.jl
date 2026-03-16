using EvoLP

# Bitvector population generator
function generate_population(pop_size::Int, chromosome_length::Int)::BitVector
    return binary_vector_pop(pop_size, chromosome_length)
end