using Plots
include("../common/parse_data.jl")
include("./calculate_optimas.jl")

println("Parsing data...")

_, landscape, _ = parse_file("train_data/01-breast-w_lr_F.h5")
#_, landscape, _ = parse_file("train_data/05-credit-a_rf_F.h5")
#_, landscape, _ = parse_file("train_data/08-letter-r_knn_F.h5")

n_combinations = length(landscape)
n_features = round(Int, log2(n_combinations + 1)) 

optima_indices = calculate_optimas(landscape)
n_optima = length(optima_indices)
println("Found $n_optima local optima!")

global_opt_idx = argmax(landscape)
global_opt_val = landscape[global_opt_idx]

distances_to_global = [count_ones(idx ⊻ global_opt_idx) for idx in optima_indices]
optima_fitnesses = landscape[optima_indices]

println("Generating Local Optima Network...")

p_lon = plot(title="Local Optima Network",
             xlabel="Hamming Distance to Global Optimum",
             ylabel="Fitness Score",
             legend=false,
             grid=true,
             size=(900, 600))


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