include("particle.jl")

mutable struct Swarm
  particles::Vector{Particle}
  global_best::BitVector
end