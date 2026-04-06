using EvoLP

function best_rank_crowding(p::Individual, q::Individual)::Bool
    p.rank < q.rank || (p.rank == q.rank && p.crowding_distance > q.crowding_distance)
end

struct NSGATournament <: ParentSelector
    T::Int
end

function select(sel::NSGATournament, y::Vector{Individual}; rng=Random.GLOBAL_RNG)
    getparent() = begin
        idxs = randperm(rng, length(y))
        best = idxs[1]
        for i in 2:sel.T
            candidate = idxs[i]
            if better_rank_crowding(y[candidate], y[best])
                best = candidate
            end
        end
        return y[best]
    end

    return [getparent(), getparent()]
end

function generate_offspring!(
    parents::Vector{Individual},
    offspring::Vector{Individual},
    SELECTOR::EvoLP.Selector,
    crossover_rate::Float64,
    mutation_rate::Float64)

    idx = 1
    while idx < pop_size
        parents = EvoLP.select(SELECTOR, parents)
    end

        # TODO: Select parents (tournament)
        # TODO: Crossover
        # TODO: Mutation
end