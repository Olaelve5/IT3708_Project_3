using Plots

function plot_fitness_history(avg_fitness::AbstractVector, best_fitness::AbstractVector)
    # Ensure both arrays are the same length
    if length(avg_fitness) != length(best_fitness)
        error("Average and best fitness arrays must be of the same length.")
    end

    generations = 1:length(avg_fitness)

    # Initialize the plot with the average fitness line
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
    
    # Add the best fitness line to the same plot
    plot!(
        p, 
        generations, 
        best_fitness, 
        label="Best Fitness", 
        linewidth=2, 
        color=:green
    )

    # Find the maximum value and the generation (index) it occurred in
    max_best_val, max_best_idx = findmax(best_fitness)

    # Add the star marker at the peak
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

    # Display the plot
    display(p)
    return p
end