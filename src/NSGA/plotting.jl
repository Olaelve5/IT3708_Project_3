using Plots
using Statistics

function plot_nsga_fitness_progression(
    dataset;
    pop_size::Int = 100,
    num_gens::Int = 100,
    tournament_size::Int = 3,
    crossover_rate::Float64 = 0.9,
    output_file::String = "nsga_accuracy_progression.png")

    accuracy_vector, _, n_features = dataset.loader()
    evaluate = make_evaluate(accuracy_vector)

    cfg = NSGAConfig(
        pop_size = pop_size,
        num_gens = num_gens,
        tournament_size = tournament_size,
        n_features = n_features,
        evaluate = evaluate,
        crossover_rate = crossover_rate,
        mutation_rate = 1 / n_features,
        log_every = num_gens,
        verbose = false
    )

    run = run_nsga(cfg)

    avg_accuracy = run.history.avg_f1_accuracy_by_gen
    best_so_far = accumulate(max, run.history.best_accuracy_by_gen)

    generations = 0:(length(avg_accuracy) - 1)
    overall_best = maximum(best_so_far)
    best_gen_idx = findfirst(==(overall_best), best_so_far)
    best_gen = isnothing(best_gen_idx) ? 0 : (best_gen_idx - 1)

    plt = plot(
        generations,
        avg_accuracy;
        label = "Average F1 Accuracy",
        color = :blue,
        linestyle = :solid,
        linewidth = 2.5,
        xlabel = "Generation",
        ylabel = "Accuracy",
        title = "Accuracy Progression over Time ($(dataset.name))",
        size = (900, 550))

    plot!(
        plt,
        generations,
        best_so_far;
        label = "Best Accuracy",
        color = :green,
        linewidth = 2)

    scatter!(
        plt,
        [best_gen],
        [overall_best];
        label = "Overall Best ($(round(overall_best, digits=6)))",
        marker = :star5,
        markersize = 12,
        color = :gold,
        markerstrokecolor = :black)

    savefig(plt, output_file)
    display(plt)
    println("Saved plot to $output_file")
    return plt
end
