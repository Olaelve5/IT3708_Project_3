using EvoLP

const CROSSOVER = EvoLP.SinglePointRecombinator()

function best_rank_crowding(p::Individual, q::Individual)::Bool
    p.rank < q.rank || (p.rank == q.rank && p.crowding_distance > q.crowding_distance)
end

struct NSGATournament <: EvoLP.ParentSelector
    T::Int
end

function select(sel::NSGATournament, y::Vector{Individual}; rng=Random.GLOBAL_RNG)
    getparent() = begin
        idxs = randperm(rng, length(y))
        best = idxs[1]
        for i in 2:sel.T
            candidate = idxs[i]
            if best_rank_crowding(y[candidate], y[best])
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
    SELECTOR::EvoLP.ParentSelector,
    crossover_rate::Float64,
    MUTATOR::EvoLP.Mutator,
    evaluate::Function)

    pop_size = length(offspring)
    idx = 1
    while idx < pop_size
        p1, p2 = EvoLP.select(SELECTOR, parents)
        if rand() < crossover_rate
            c1_genome = EvoLP.cross(CROSSOVER, p1.genome, p2.genome)
            c2_genome = EvoLP.cross(CROSSOVER, p1.genome, p2.genome)
            c1 = new_individual(c1_genome, evaluate(c1_genome))
            c2 = new_individual(c2_genome, evaluate(c2_genome))
        else
            c1, c2 = deepcopy(p1), deepcopy(p2)
        end
        offspring[idx], offspring[idx+1] = c1, c2
        idx += 2
    end

    # Mutate all offspring with mutation_rate chance per gene
    for p in offspring
        p.genome = EvoLP.mutate(MUTATOR, p.genome)
        p.accuracy = evaluate(p.genome)
        p.num_features = count_ones(p.genome)
    end
end