using HDF5
using Statistics # Needed for the mean() function

# 1. Read the data
file = h5open("train_data/01-breast-w_lr_F.h5", "r")
accuracy_matrix = read(file["accuracies"])
close(file)

# 2. Set up the parameters
epsilon = 0.01 # A small constant between 0 and 1 for the penalty
num_combinations = size(accuracy_matrix, 1)

# Initialize the 1D array to store the final fitness values
fitness_landscape = zeros(Float64, num_combinations)

# 3. Calculate fitness for each combination
for i in 1:num_combinations
    # h_a: Mean accuracy across the columns [cite: 90]
    h_a = mean(accuracy_matrix[i, :])
    
    # h_p: The penalty. The row index 'i' is the decimal representation 
    # of the binary feature mask. count_ones() counts the active features.
    h_p = count_ones(i) 
    
    # Calculate final fitness 
    fitness_landscape[i] = h_a - (epsilon * h_p)
end

println("Successfully built fitness landscape with ", length(fitness_landscape), " combinations!")


# 1. Look at the raw data (the accuracy matrix)
println("--- RAW ACCURACY MATRIX (First 5 combinations) ---")
# display() is great in Julia for showing matrices neatly
display(accuracy_matrix[1:5, :]) 

println("\n")

# 2. Look at your final processed data (the 1D lookup table)
println("--- CALCULATED FITNESS LANDSCAPE (First 5 combinations) ---")
display(fitness_landscape[1:5])