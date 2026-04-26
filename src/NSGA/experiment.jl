using Statistics

"""
    run_nsga_experiment(cfg::NSGAConfig, num_runs::Int; dataset_name::String="Unknown")

Run NSGA-II multiple times and collect statistics for Step 5 comparison.
Returns a NamedTuple with aggregated results suitable for tables/reports.
"""
function run_nsga_experiment(cfg::NSGAConfig, num_runs::Int; dataset_name::String="Unknown")
    # Create quiet config (disable verbose output for batch runs)
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
    all_best_acc_histories = Vector{Vector{Float64}}()
    
    println("Running NSGA-II $num_runs times on '$dataset_name'...")
    
    for run_idx in 1:num_runs
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
    
    # Compute aggregate statistics
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
        avg_best_acc_by_gen = avg_best_acc_by_gen
    )
end

"""
    compute_stats(values::Vector)

Compute mean, std, min, max, median for a vector of values.
"""
function compute_stats(values::AbstractVector{<:Real})
    return (
        mean = mean(values),
        std = std(values),
        min = minimum(values),
        max = maximum(values),
        median = median(values)
    )
end

"""
    average_histories(histories::Vector{Vector})

Average a list of history vectors to same length.
Pads shorter histories with their last value if needed.
"""
function average_histories(histories::AbstractVector{<:AbstractVector{<:Real}})
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

"""
    print_experiment_summary(result::NamedTuple)

Print a formatted summary of experiment results for Step 5.
"""
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

"""
    export_results_table(results::Vector{NamedTuple}; filename::String="nsga_results.tex", caption::String="NSGA-II comparison across datasets", label::String="tab:nsga_results")

Export comparison table of multiple experiment results as a LaTeX table.
"""
function export_results_table(
    results::Vector{NamedTuple};
    filename::String="nsga_results.tex",
    caption::String="NSGA-II comparison across datasets",
    label::String="tab:nsga_results")

    latex_escape(s::AbstractString) = replace(s, "_" => "\\_")

    open(filename, "w") do f
        write(f, "\\begin{table}[ht]\n")
        write(f, "\\centering\n")
        write(f, "\\caption{$(latex_escape(caption))}\n")
        write(f, "\\label{$(latex_escape(label))}\n")
        write(f, "\\begin{tabular}{lrrrrr}\n")
        write(f, "\\hline\n")
        write(f, "Dataset & Runs & F1 Mean & F1 Std & Best Acc Mean & Best Acc Std \\\\ \n")
        write(f, "\\hline\n")

        for r in results
            dataset = latex_escape(r.dataset_name)
            f1_mean = round(r.f1_size.mean, digits=2)
            f1_std = round(r.f1_size.std, digits=2)
            acc_mean = round(r.best_accuracy.mean, digits=6)
            acc_std = round(r.best_accuracy.std, digits=6)
            write(f, "$dataset & $(r.num_runs) & $f1_mean & $f1_std & $acc_mean & $acc_std \\\\ \n")
        end

        write(f, "\\hline\n")
        write(f, "\\end{tabular}\n")
        write(f, "\\end{table}\n")
    end
    println("Results exported to $filename")
end

function run_step5_experiments_nsga(datasets; num_runs::Int)
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
            verbose = true,
            log_every = 10
        )

        result = run_nsga_experiment(nsga_cfg, num_runs; dataset_name = dataset_name)
        push!(experiment_results, result)
        print_experiment_summary(result)
    end

    export_results_table(experiment_results; filename = "nsga_comparison_results.txt")
    println("Comparison table saved to nsga_comparison_results.txt")
end