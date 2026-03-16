using HDF5
using Statistics

function parse_file(filepath)
    file = h5open(filepath, "r")
    accuracy_matrix = read(file["accuracies"])
    close(file)

    epsilon = 0.01
    num_combinations = size(accuracy_matrix, 1)

    # 1D array to store the final fitness values
    fitness_landscape = zeros(Float64, num_combinations)

    for i in 1:num_combinations
        h_a = mean(accuracy_matrix[i, :])
        h_p = count_ones(i) 
        fitness_landscape[i] = h_a - (epsilon * h_p)
    end
    
    return fitness_landscape
end