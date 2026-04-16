include("common/parse_data.jl")
include("common/evaluation.jl")
include("common/config.jl")
include("NSGA/nsga.jl")
include("PSO/pso.jl")
include("./common/config.jl")
include("./common/generate_synthetic.jl")

using .PSO

# Parameters
data_path = "./train_data/01-breast-w_lr_F.h5" # 01-breast-w_lr_F.h5 | 05-credit-a_rf_F.h5 | 08-letter-r_knn_F.h5

# Get fitness landscape and number of instance features from dataset
#accuracy_vector, fitness_landscape, n_features = parse_file(data_path)
accuracy_vector, fitness_landscape, n_features = generate_synthetic_fitness_landscape()

# Make and evaluate function that remembers fitness landscape
evaluate = make_evaluate(fitness_landscape)
nsga_evaluate =  make_evaluate(accuracy_vector)

# Instantiate config files with instance specifics
nsga_cfg = NSGAConfig(n_features = n_features, evaluate = nsga_evaluate)
pso_cfg = PSOConfig(
    num_particles = 100,
    num_features = n_features,
    num_iterations = 1000,
    inertia_weight = 0.7,
    cognitive_coefficient = 1.5,
    social_coefficient = 1.5,
    evaluate = evaluate
)
# TODO: PSO config
# TODO: SGA config

#run_nsga(nsga_cfg)
# TODO: Call PSO
best_features = run_pso(pso_cfg)
println("Best feature mask: ", best_features)
println("Number of features selected: ", sum(best_features))
println("Fitness: ", pso_cfg.evaluate(best_features))
# TODO: Call SGA