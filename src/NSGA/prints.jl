using Statistics

function rank1_front(population::Vector{Individual})::Vector{Individual}
	return [p for p in population if p.rank == 1]
end

function best_solution(front::Vector{Individual})::Individual
	# Prefer highest accuracy and break ties by fewer features.
	return sort(front, by = p -> (-p.accuracy, p.num_features))[1]
end

function front_stats(front::Vector{Individual})
	if isempty(front)
		return (size = 0, best_accuracy = NaN, mean_accuracy = NaN, min_features = 0, median_features = NaN)
	end

	best_accuracy = maximum(p.accuracy for p in front)
	mean_accuracy = mean(p.accuracy for p in front)
	min_features = minimum(p.num_features for p in front)
	median_features = median(p.num_features for p in front)
	return (
		size = length(front),
		best_accuracy = best_accuracy,
		mean_accuracy = mean_accuracy,
		min_features = min_features,
		median_features = median_features	)
end

function print_run_header(cfg::NSGAConfig)
	println("--- NSGA-II Run ---")
	println("Population Size: $(cfg.pop_size)")
	println("Generations: $(cfg.num_gens)")
	println("Mutation Rate: $(cfg.mutation_rate)")
	println("Crossover Rate: $(cfg.crossover_rate)")
	println("Tournament Size: $(cfg.tournament_size)")
	println("Log Every: $(cfg.log_every)")
end

function print_generation_summary(label::String, gen::Int, front::Vector{Individual})
	s = front_stats(front)
	println(
		"[$label] Gen=$gen | F1=$(s.size) | BestAcc=$(round(s.best_accuracy, digits=6)) | " *
		"MinFeat=$(s.min_features) | MedianFeat=$(round(s.median_features, digits=3))")
end

function print_final_summary(front::Vector{Individual})
	if isempty(front)
		println("[NSGA-II] Final front is empty.")
		return
	end

	best = representative_solution(front)
	println("--- NSGA-II Final ---")
	println("Final F1 size: $(length(front))")
	println("Representative solution | accuracy=$(round(best.accuracy, digits=6)) | features=$(best.num_features)")
end