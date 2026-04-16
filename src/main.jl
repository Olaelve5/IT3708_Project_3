include("common/parse_data.jl")
include("common/evaluation.jl")
include("common/config.jl")
include("common/population.jl")
include("NSGA/nsga.jl")
include("pso/pso.jl")
include("./common/config.jl")
include("./common/generate_synthetic.jl")

# Parameters
data_path = "./train_data/01-breast-w_lr_F.h5" # 01-breast-w_lr_F.h5 | 05-credit-a_rf_F.h5 | 08-letter-r_knn_F.h5

# Get fitness landscape and number of instance features from dataset
accuracy_vector, fitness_landscape, n_features = parse_file(data_path)
#accuracy_vector, fitness_landscape, n_features = generate_synthetic_fitness_landscape()

# Make an evaluate function that remembers fitness landscape
evaluate = make_evaluate(fitness_landscape)
nsga_evaluate =  make_evaluate(accuracy_vector)

# Instantiate config files with instance specifics
nsga_cfg = NSGAConfig(n_features = n_features, evaluate = nsga_evaluate, mutation_rate = 1/n_features)
pso_cfg = PSOConfig(
    num_particles = 1000,
    num_features = n_features,
    num_iterations = 10000,
    inertia_weight = 1,
    cognitive_coefficient = 1,
    social_coefficient = 1,
    evaluate = evaluate
)
# TODO: SGA config

nsga_run = run_nsga(nsga_cfg)
println("NSGA-II returned $(length(nsga_run.pareto_front)) Pareto-front solutions.")
# TODO: Call PSO
# best_features = run_pso(pso_cfg)
# println("Best feature mask: ", best_features)
# println("Number of features selected: ", sum(best_features))
# println("Fitness: ", pso_cfg.evaluate(best_features))
# TODO: Call SGA