
function front_crowding_distance!(front::Vector{Individual})
    # Get pop size
    pop_size = length(front)

    # Reset crowding distances to 0
    for p in front
        p.crowding_distance = 0.0
    end

    # For each fitness function
    for get_field in (p -> p.accuracy, p -> p.num_features)
        # Sort by fitness
        sorted = sort(front, by = get_field)
        range = get_field(sorted[end]) - get_field(sorted[1])

        # Avoid division by zero if all individuals in front have same value
        range == 0 && continue

        # Set edges crowding distance to Inf
        sorted[1].crowding_distance = sorted[end].crowding_distance = Inf
        for i in 2:pop_size-1
            sorted[i].crowding_distance += (get_field(sorted[i+1]) - get_field(sorted[i-1])) / range
        end
    end
end


function crowding_distance!(population::Vector{Individual})
    # Get highest rank
    rank_max = maximum(p.rank for p in population)
    
    # Create an empty vector of vectors of individuals
    fronts = [Individual[] for _ in 1:rank_max]

    # Assign population to correct rank vector
    for p in population
        push!(fronts[p.rank], p)
    end

    # Assign crowding distance to population by front
    for front in fronts
        front_crowding_distance!(front)
    end
end