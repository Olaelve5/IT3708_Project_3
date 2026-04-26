using Plots
include("../common/parse_data.jl")
include("./calculate_optimas.jl")
include("../common/evaluation.jl")
include("../common/generate_synthetic.jl")

println("Parsing data...")

#_, landscape, _ = parse_file("train_data/01-breast-w_lr_F.h5")
landscape, _, _ = parse_file("train_data/05-credit-a_rf_F.h5")
#_, landscape, _ = parse_file("train_data/08-letter-r_knn_F.h5")
#_, landscape, _ = parse_file("test_data/10-hepatitis_lr_F.h5", epsilon=0)
#_, landscape, _ = parse_file("test_data/06-zoo_lr_F.h5", epsilon=0)
#acc_vec, landscape, N_FEATURES = generate_test_synthetic()
#acc_vec, landscape, N_FEATURES = generate_train_synthetic()

n_combinations = length(landscape)
n_features = round(Int, log2(n_combinations + 1)) 

all_optima_indices = calculate_optimas(landscape, n_features)[3]
total_optima = length(all_optima_indices)
println("Found $total_optima local optima!")

sorted_order = sortperm(landscape[all_optima_indices], rev=true)
sorted_optima_indices = all_optima_indices[sorted_order]

n_optima = min(4000, total_optima)
step_indices = round.(Int, range(1, total_optima, length=n_optima))
optima_indices = sorted_optima_indices[step_indices]

println("Filtering to a representative sample of $n_optima optima for the network plot...")

global_opt_idx = argmax(landscape)
global_opt_val = landscape[global_opt_idx]

distances_to_global = [count_ones(idx ⊻ global_opt_idx) for idx in optima_indices]
optima_fitnesses = landscape[optima_indices]

println("Generating Local Optima Network...")
min_y = minimum(optima_fitnesses)
max_y = maximum(optima_fitnesses)

y_padding = max(0.05, (max_y - min_y) * 0.1) 

p_lon = plot(title="Local Optima Network (Cross-Section of $n_optima)",
             xlabel="Hamming Distance to Global Optimum",
             ylabel="Fitness Score",
             legend=false,
             grid=true,
             size=(900, 600),
             ylims=(min_y - y_padding, max_y + y_padding))


max_jump_distance = 3 

for i in 1:n_optima
    for j in (i+1):n_optima
        # Calculate Hamming distance between the two peaks
        dist_between_peaks = count_ones(optima_indices[i] ⊻ optima_indices[j])
        
        if dist_between_peaks <= max_jump_distance
            plot!(p_lon, 
                  [distances_to_global[i], distances_to_global[j]], 
                  [optima_fitnesses[i], optima_fitnesses[j]],
                  linecolor=:gray, 
                  linewidth=1, 
                  alpha=0.4,
                  label="")
        end
    end
end

scatter!(p_lon, distances_to_global, optima_fitnesses, 
         color=:orange, 
         markersize=5, 
         markerstrokewidth=1,
         markerstrokecolor=:white,
         label="Local Optima")

scatter!(p_lon, [0], [global_opt_val], 
         color=:red, 
         markersize=10, 
         marker=:circle, 
         label="Global Optimum", 
         legend=true)

display(p_lon)

println("Press Enter to close the plot...")
readline()