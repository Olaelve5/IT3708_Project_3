include("common/parse_data.jl")
include("common/evaluation.jl")
include("common/config.jl")
include("common/population.jl")
include("NSGA/nsga.jl")
include("pso/pso.jl")
include("./common/config.jl")
include("./common/generate_synthetic.jl")

# Parameters
data_path = "./train_data/05-credit-a_rf_F.h5" # 01-breast-w_lr_F.h5 | 05-credit-a_rf_F.h5 | 08-letter-r_knn_F.h5

# Get fitness landscape and number of instance features from dataset
accuracy_vector, fitness_landscape, n_features = parse_file(data_path)
#accuracy_vector, fitness_landscape, n_features = generate_synthetic_fitness_landscape()

# Make an evaluate function that remembers fitness landscape
evaluate = make_evaluate(fitness_landscape)
nsga_evaluate =  make_evaluate(accuracy_vector)
optimal_fitness = maximum(fitness_landscape)

# Instantiate config files with instance specifics
nsga_cfg = NSGAConfig(n_features = n_features, evaluate = nsga_evaluate, mutation_rate = 1/n_features)
pso_cfg = PSOConfig(
    num_particles = 5,
    num_features = n_features,
    num_iterations = 100,
    inertia_weight = 1,
    cognitive_coefficient = 1,
    social_coefficient = 1,
    evaluate = evaluate,
    log_every = 10,
    verbose = true
)
# TODO: SGA config

nsga_run = run_nsga(nsga_cfg)
println("NSGA-II returned $(length(nsga_run.pareto_front)) Pareto-front solutions.")

pso_run = PSO.run_pso(pso_cfg)
println("PSO best feature mask: ", pso_run.best_solution)
println("PSO number of features selected: ", sum(pso_run.best_solution))
println("PSO best fitness: ", pso_run.best_fitness)
println("PSO best solution first found at iteration: ", pso_run.best_iteration)

pso_experiment = PSO.run_pso_experiment(pso_cfg, 10; dataset_name = data_path, optimal_fitness = optimal_fitness)
println("PSO over $(pso_experiment.num_runs) runs:")
println("  Best fitness mean: ", round(pso_experiment.best_fitness.mean, digits = 6))
println("  Best fitness std:  ", round(pso_experiment.best_fitness.std, digits = 6))
println("  Best iteration mean: ", round(pso_experiment.best_iteration.mean, digits = 2))
println("  Best iteration std:  ", round(pso_experiment.best_iteration.std, digits = 2))
println("  Optimal fitness: ", round(optimal_fitness, digits = 6))
println("  Success count: ", pso_experiment.success.count, "/", pso_experiment.num_runs)
println("  Success rate: ", round(100 * pso_experiment.success.rate, digits = 2), "%")
# TODO: Call SGA
