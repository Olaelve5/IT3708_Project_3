using Plots

fitness_values = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 6]
active_bits = 0:31

p = plot(active_bits, fitness_values, 
    xlabel = "Number of Active Features", 
    ylabel = "Fitness", 
    legend = false, 
    linewidth = 2, 
    color = :blue,
    markershape = :circle, 
    markercolor = :blue,
    markersize = 3,
    markerstrokewidth = 0,
    xticks = 0:5:31, 
    yticks = 0:1:6,
    grid = true,
    size = (800, 400)
)


display(p)
println("Plot generated. Press Enter in the terminal to close...")
readline()