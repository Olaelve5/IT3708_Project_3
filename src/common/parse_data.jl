using HDF5
using Statistics

function parse_file(filepath; epsilon=0.1)
    file = h5open(filepath, "r")
    accuracy_matrix = read(file["accuracies"])
    close(file)

    num_combinations = size(accuracy_matrix, 1)
    num_features = log2(num_combinations+1)

    # 1D array to store the final fitness values
    fitness_landscape = Vector{Float64}(undef, num_combinations)
    accuracy_vector = Vector{Float64}(undef, num_combinations)


    for i in 1:num_combinations
        h_a = mean(accuracy_matrix[i, :])
        accuracy_vector[i] = h_a
        h_p = count_ones(i) 
        fitness_landscape[i] = h_a - (epsilon * h_p)
    end
    
    return accuracy_vector, fitness_landscape, num_features
end