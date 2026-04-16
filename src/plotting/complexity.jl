using Plots
include("../common/parse_data.jl")

println("Parsing data...")
_, landscape, _ = parse_file("train_data/01-breast-w_lr_F.h5")
#_, landscape, _ = parse_file("train_data/05-credit-a_rf_F.h5")
#_, landscape, _ = parse_file("train_data/08-letter-r_knn_F.h5")
n_combinations = length(landscape)

feature_counts = Int[]
fitnesses = Float64[]

for i in 1:n_combinations
    # count_ones(i) gives us exactly how many features are active in this combination
    push!(feature_counts, count_ones(i))
    push!(fitnesses, landscape[i])
end

println("Generating Complexity Plot...")
p_complexity = scatter(feature_counts, fitnesses,
                       alpha=0.15, 
                       color=:teal,
                       markersize=3,
                       markerstrokewidth=0,
                       xlabel="Number of Active Features",
                       ylabel="Fitness (Accuracy - Penalty)",
                       title="Fitness vs. Model Complexity",
                       legend=false,
                       xticks=0:1:16,
                       size=(800, 500))

global_opt_idx = argmax(landscape)
global_opt_val = landscape[global_opt_idx]
global_opt_features = count_ones(global_opt_idx)

scatter!(p_complexity, [global_opt_features], [global_opt_val], 
         color=:red, markersize=8, marker=:circle, label="Global Optimum", legend=true)

display(p_complexity)

println("Press Enter to close the plot...")
readline()