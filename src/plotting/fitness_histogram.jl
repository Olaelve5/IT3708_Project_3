using Plots
include("../parse_data.jl")

println("Parsing data...")
#landscape = parse_file("train_data/01-breast-w_lr_F.h5")
landscape = parse_file("train_data/05-credit-a_rf_F.h5")
#landscape = parse_file("train_data/08-letter-r_knn_F.h5")

println("Generating Histogram...")
p_hist = histogram(landscape, 
                   bins=100,
                   color=:indigo,
                   linecolor=:white,
                   xlabel="Fitness Score",
                   ylabel="Number of Combinations",
                   title="Distribution of Fitness Values",
                   legend=false,
                   size=(800, 500))

display(p_hist)

println("Press Enter to close the plot...")
readline()