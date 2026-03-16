struct PSOConfig
    num_particles::Int
    num_features::Int
    num_iterations::Int
    inertia_weight::Float64
    cognitive_coefficient::Float64
    social_coefficient::Float64
end

function PSOConfig(;
  num_particles = 30,
  num_features, num_iterations = 30,
  inertia_weight = 0.7,
  cognitive_coefficient = 1.5,
  social_coefficient = 1.5 
  )
  @assert n_particles > 0 "num_particles must be positive"
  @assert n_features > 0 "num_features must be positive"
  @assert 0 < inertia < 1 "inertia_weight must be between 0 and 1"
  PSOConfig(num_particles, num_features, num_iterations, inertia_weight, cognitive_coefficient, social_coefficient)
end