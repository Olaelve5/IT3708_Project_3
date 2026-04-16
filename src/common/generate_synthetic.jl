function generate_synthetic_fitness_landscape(;m::Int = 1, s::Int = 4, n::Int = 16, epsilon = 0.1)
  function g(b::Int)
    if b % s == 0
            return m * s
        else
            return m * (b % s)
        end
  end

  function triangle(b::Int)
    if ceil(b / s) % 2 == 1
            return g(b)
        else
            return m * (ceil(Int, b / s) * s - b)
        end
  end
  accuracy_vector = [Float64(triangle(count_ones(i))) for i in 0:2^n]
  fitness_landscape = [Float64(triangle(count_ones(i))) - epsilon * count_ones(i) for i in 0:2^n]
  num_features = n

  return accuracy_vector, fitness_landscape, num_features
end 