using Plots
include("../common/parse_data.jl")

println("Parsing data...")
#_, landscape, _ = parse_file("train_data/01-breast-w_lr_F.h5")
#_, landscape, _ = parse_file("train_data/05-credit-a_rf_F.h5")
#_, landscape, _ = parse_file("train_data/08-letter-r_knn_F.h5")

_, landscape, _ = parse_file("test_data/10-hepatitis_lr_F.h5", epsilon=0)
#_, landscape, _ = parse_file("test_data/06-zoo_lr_F.h5", epsilon=0)

println("Calculating Fitness-Distance Correlation...")
n_combinations = length(landscape)

# find global optimum
global_opt_idx = argmax(landscape)
global_opt_fitness = landscape[global_opt_idx]

println("Global Optimum found at index $global_opt_idx with fitness $global_opt_fitness")

distances = Int[]
fitnesses = Float64[]

for i in 1:n_combinations
    # compute Hamming distance using XOR and count_ones
    dist = count_ones(i ⊻ global_opt_idx)
    
    push!(distances, dist)
    push!(fitnesses, landscape[i])
end

println("Generating plot...")

p2 = scatter(distances, fitnesses, 
             alpha=0.15,               
             color=:blue, 
             markersize=3, 
             markerstrokewidth=0,
             xlabel="Hamming Distance to Global Optimum", 
             ylabel="Fitness (Accuracy - Penalty)", 
             title="Fitness-Distance Scatter Plot",
             legend=false,
             size=(800, 500))

scatter!(p2, [0], [global_opt_fitness], 
         color=:red, markersize=8, marker=:circle, label="Global Optimum", legend=true)

display(p2)

println("Press Enter in the terminal to close the plot and exit.")
readline()