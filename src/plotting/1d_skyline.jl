using Plots
include("../common/parse_data.jl")
include("./calculate_optimas.jl")

_, landscape, _ = parse_file("train_data/01-breast-w_lr_F.h5")
#_, landscape, _ = parse_file("train_data/05-credit-a_rf_F.h5")
#_, landscape, _ = parse_file("train_data/08-letter-r_knn_F.h5")

n_combinations = length(landscape)
n_features = round(Int, log2(n_combinations + 1)) 
optima_indices = calculate_optimas(landscape)

println("Found $(length(optima_indices)) local optima!")

println("Generating plot...")
p1 = plot(landscape, label="Fitness", linecolor=:gray, alpha=0.6,
          xlabel="Genotype (Decimal Index)", ylabel="Fitness",
          title="1D Skyline Plot of Fitness Landscape")

scatter!(p1, optima_indices, landscape[optima_indices], 
         color=:orange, markersize=4, markerstrokewidth=0, label="Local Optima")

display(p1)
println("Press Enter in the terminal to close the plot and exit.")
readline()