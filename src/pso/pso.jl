module PSO

using Main: PSOConfig
using Random: bitrand
using Statistics: mean, std

include("particle.jl")
include("swarm.jl")
include("algorithm.jl")

export PSORunResult, run_pso, run_pso_experiment

end
