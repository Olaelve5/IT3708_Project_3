using Statistics

function run_nsga_experiment(cfg::NSGAConfig, num_runs::Int; dataset_name::String="Unknown")
    # Quiet config sets verbose to false for multiple runs/datasets
    cfg_quiet = NSGAConfig(
        pop_size = cfg.pop_size,
        num_gens = cfg.num_gens,
        tournament_size = cfg.tournament_size,
        n_features = cfg.n_features,
        evaluate = cfg.evaluate,
        crossover_rate = cfg.crossover_rate,
        mutation_rate = cfg.mutation_rate,
        log_every = cfg.log_every,
        verbose = false
    )
    
    # Storage for results across all runs
    all_f1_sizes = Int[]
    all_best_accuracies = Float64[]
    all_final_pareto_fronts = Vector{Vector{Individual}}()
    all_final_representatives = Vector{Individual}()
    all_f1_histories = Vector{Vector{Int}}()
    all_best_acc_histories = Vector{Vector{Real}}()
    
    println("Running NSGA-II $num_runs times on '$dataset_name'...")
    
    for _ in 1:num_runs
        result = run_nsga(cfg_quiet)
        
        push!(all_f1_sizes, length(result.pareto_front))
        push!(all_best_accuracies, result.representative.accuracy)
        push!(all_final_pareto_fronts, result.pareto_front)
        push!(all_final_representatives, result.representative)
        push!(all_f1_histories, result.history.f1_size_by_gen)
        push!(all_best_acc_histories, result.history.best_accuracy_by_gen)
        
        print(".")
    end
    println()
    
    # Compute aggregated statistics
    f1_size_stats = compute_stats(all_f1_sizes)
    best_acc_stats = compute_stats(all_best_accuracies)
    
    # Collect convergence curves (average over all runs at each generation)
    avg_f1_by_gen = average_histories(all_f1_histories)
    avg_best_acc_by_gen = average_histories(all_best_acc_histories)
    
    return (
        dataset_name = dataset_name,
        num_runs = num_runs,
        f1_size = f1_size_stats,
        best_accuracy = best_acc_stats,
        all_representatives = all_final_representatives,
        avg_f1_by_gen = avg_f1_by_gen,
        avg_best_acc_by_gen = avg_best_acc_by_gen)
end

function compute_stats(values::Vector{Real})
    return (
        mean = mean(values),
        std = std(values),
        min = minimum(values),
        max = maximum(values),
        median = median(values))
end

function average_histories(histories::Vector{Vector{Real}})
    if isempty(histories)
        return Float64[]
    end
    
    max_len = maximum(length(h) for h in histories)
    
    avg = Float64[]
    for i in 1:max_len
        vals = Float64[]
        for h in histories
            if i <= length(h)
                push!(vals, h[i])
            else
                # Use last value for padded positions
                push!(vals, h[end])
            end
        end
        push!(avg, mean(vals))
    end
    
    return avg
end

function print_experiment_summary(result::NamedTuple)
    println("\n" * "="^80)
    println("NSGA-II EXPERIMENT SUMMARY: $(result.dataset_name)")
    println("="^80)
    println("Number of runs: $(result.num_runs)")
    println()
    
    println("Pareto Front Size (F1):")
    println("  Mean:   $(round(result.f1_size.mean, digits=2))")
    println("  Std:    $(round(result.f1_size.std, digits=2))")
    println("  Min:    $(result.f1_size.min)")
    println("  Max:    $(result.f1_size.max)")
    println("  Median: $(result.f1_size.median)")
    println()
    
    println("Best Accuracy in F1:")
    println("  Mean:   $(round(result.best_accuracy.mean, digits=6))")
    println("  Std:    $(round(result.best_accuracy.std, digits=6))")
    println("  Min:    $(round(result.best_accuracy.min, digits=6))")
    println("  Max:    $(round(result.best_accuracy.max, digits=6))")
    println("  Median: $(round(result.best_accuracy.median, digits=6))")
    println()
    
    println("Representative Solution (best accuracy across all runs):")
    rep = sort(result.all_representatives, by = p -> (-p.accuracy, p.num_features))[1]
    println("  Accuracy: $(round(rep.accuracy, digits=6))")
    println("  Features: $(rep.num_features)")
    println("  Genome:   $(rep.genome)")
    println("="^80 * "\n")
end

function run_multiple_experiments_nsga(datasets; num_runs::Int)
    experiment_results = NamedTuple[]

    for dataset in datasets
        dataset_name = dataset.name
        println("\n" * "="^80)
        println("Processing: $dataset_name")
        println("="^80)

        accuracy_vector, _, n_features = dataset.loader()
        println("Max accuracy for $dataset_name: $(maximum(accuracy_vector))")
        nsga_evaluate = make_evaluate(accuracy_vector)

        nsga_cfg = NSGAConfig(
            n_features = n_features,
            evaluate = nsga_evaluate,
            mutation_rate = 1 / n_features,
            verbose = false,
            log_every = 10
        )

        result = run_nsga_experiment(nsga_cfg, num_runs; dataset_name = dataset_name)
        push!(experiment_results, result)
        print_experiment_summary(result)
    end
end