module PSO

using Main: PSOConfig
using Random: bitrand

include("particle.jl")
include("swarm.jl")
include("algorithm.jl")

export run_pso

end