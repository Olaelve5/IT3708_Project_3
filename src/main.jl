include("common/parse_data.jl")
include("common/evaluation.jl")
include("NSGA/nsga.jl")

# Parameters
data_path = "./01-breast-w_lr_F.h5" # 01-breast-w_lr_F.h5 | 05-credit-a_rf_F.h5 | 08-letter-r_knn_F.h5

# Get fitness landscape and number of instance features from dataset
fitness_landscape, n_features = parse_file(data_path)

# Make and evaluate function that remembers fitness landscape
evaluate = make_evaluate(fitness_landscape)

# Instantiate config files with instance specifics
nsga_cfg = NSGAConfig(n_features = n_features, evaluate = evaluate)
# TODO: PSO config
# TODO: SGA config

run_nsga(nsga_cfg)
# TODO: Call PSO
# TODO: Call SGA