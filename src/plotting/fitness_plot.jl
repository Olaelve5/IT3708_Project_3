using Plots

function plot_fitness_history(avg_fitness::AbstractVector, best_fitness::AbstractVector)
    if length(avg_fitness) != length(best_fitness)
        error("Average and best fitness arrays must be of the same length.")
    end

    generations = 1:length(avg_fitness)

    p = plot(
        generations, 
        avg_fitness, 
        label="Average Fitness", 
        linewidth=2, 
        color=:blue, 
        xlabel="Generation", 
        ylabel="Fitness", 
        title="Fitness Progression over Time",
        legend=:bottomright
    )
    
    plot!(
        p, 
        generations, 
        best_fitness, 
        label="Best Fitness", 
        linewidth=2, 
        color=:green
    )

    max_best_val, max_best_idx = findmax(best_fitness)

    scatter!(
        p, 
        [max_best_idx], 
        [max_best_val], 
        label="Overall Best ($max_best_val)", 
        markershape=:star, 
        markersize=10, 
        markercolor=:gold,
        markerstrokecolor=:black
    )

    display(p)
    return p
end