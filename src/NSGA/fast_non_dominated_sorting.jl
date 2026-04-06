

function dominates(A::Individual, B::Individual)::Bool
    better_or_equal_all = (A.accuracy >= B.accuracy) && (A.num_features <= B.num_features)
    strictly_better_one = (A.accuracy > B.accuracy) || (A.num_features < B.num_features)
    return better_or_equal_all && strictly_better_one
end

function fnd_sort!(population::Vector{Individual}, pop_size::Int)
    # Reset values from previous sorts, if any.
    for p in population
        p.n_p = 0
        empty!(p.S_p)
    end

    Fi = Individual[]
    count = 0
    for p in population
        np = 0
        for q in population
            p === q && continue
            if dominates(p, q)
                push!(p.S_p, q)
            elseif dominates(q, p)
                np += 1
            end
        end
        p.n_p = np
        if np == 0
            p.rank = 1
            push!(Fi, p)
        end
    end
    count += length(Fi)
    i = 2
    while !isempty(Fi) && count < pop_size
        Q = Individual[]
        for p in Fi
            Sp = p.S_p
            for q in Sp
                q.n_p -= 1
                if q.n_p == 0
                    q.rank = i
                    push!(Q, q)
                end
            end
        end
        i += 1
        Fi = Q
        count += length(Q)
    end
end