include("common/parse_data.jl")
include("common/evaluation.jl")
include("common/config.jl")
include("common/population.jl")
include("NSGA/experiment.jl")
include("NSGA/nsga.jl")
include("NSGA/plotting.jl")
include("pso/pso.jl")
include("./common/generate_synthetic.jl")
include("./plotting/fitness_plot.jl")

# Parameters
data_path = "./train_data/05-credit-a_rf_F.h5" # 01-breast-w_lr_F.h5 | 05-credit-a_rf_F.h5 | 08-letter-r_knn_F.h5
#data_path = "./10-hepatitis_lr_F.h5"

# Get fitness landscape and number of instance features from dataset
accuracy_vector, fitness_landscape, n_features = parse_file(data_path)
#accuracy_vector, fitness_landscape, n_features = generate_synthetic_fitness_landscape()
#accuracy_vector, fitness_landscape, n_features = generate_test_synthetic()

# Make an evaluate function that remembers fitness landscape
evaluate = make_evaluate(fitness_landscape)
#nsga_evaluate =  make_evaluate(accuracy_vector)
optimal_fitness = maximum(fitness_landscape)
#optimal_fitness = maximum(fitness_landscape)
println(optimal_fitness)

# Instantiate config files with instance specifics
pso_cfg = PSOConfig(
    num_particles = 100,
    num_features = n_features,
    num_iterations = 500,
    inertia_weight = 0.7,
    cognitive_coefficient = 1.5,
    social_coefficient = 1.5,
    evaluate = evaluate,
    log_every = 10,
    verbose = true
)


pso_experiment = PSO.run_pso_experiment(pso_cfg, 10; dataset_name = data_path, optimal_fitness = optimal_fitness)
println("PSO over $(pso_experiment.num_runs) runs:")
println("  Best fitness mean: ", round(pso_experiment.best_fitness.mean, digits = 6))
println("  Best fitness std:  ", round(pso_experiment.best_fitness.std, digits = 6))
println("  Best iteration mean: ", round(pso_experiment.best_iteration.mean, digits = 2))
println("  Best iteration std:  ", round(pso_experiment.best_iteration.std, digits = 2))
println("  Optimal fitness: ", round(optimal_fitness, digits = 6))
println("  Success count: ", pso_experiment.success.count, "/", pso_experiment.num_runs)
println("  Success rate: ", round(100 * pso_experiment.success.rate, digits = 2), "%")
println("  Best ever fitness: ", round(pso_experiment.best_ever.fitness, digits = 6))
println("  Best ever genome: ", pso_experiment.best_ever.genome)
println("  Best ever bitstring: ", pso_experiment.best_ever.bitstring)

# Datasets to run NSGA on, comment out unwanted
datasets = [
    (name = "06-zoo_lr_F", loader = () -> parse_file("test_data/06-zoo_lr_F.h5")),
    (name = "10-hepatitis_lr_F", loader = () -> parse_file("test_data/10-hepatitis_lr_F.h5")),
    (name = "01-breast-w", loader = () -> parse_file("train_data/01-breast-w_lr_F.h5")),
    (name = "05-credit-a", loader = () -> parse_file("train_data/05-credit-a_rf_F.h5")),
    (name = "08-letter-r", loader = () -> parse_file("train_data/08-letter-r_knn_F.h5")),

    # Two synthetic landscapes from assignment material
    (name = "synthetic-triangle-train", loader = () -> generate_synthetic_fitness_landscape(m = 1, s = 4, n = 16, epsilon = 0.01)),
    (name = "synthetic-triangle-test", loader = () -> generate_step6_asymmetric_synthetic_landscape()),
]

# NSGA plotting:
# plot_nsga_fitness_progression(datasets[7]; num_gens = 100)

# Run multiple nsga and gather statistics:
# run_multiple_experiments_nsga(datasets; num_runs = 10)
