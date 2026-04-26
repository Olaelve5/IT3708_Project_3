function generate_train_synthetic(;m::Int = 1, s::Int = 4, n::Int = 16, epsilon = 0.01)
  function g(b::Int)
    if b % s == 0
      return m * s
    end
    return m * (b % s)
  end

  function triangle(b::Int)
    segment = cld(b, s)
    if segment % 2 == 1
      return g(b)
    end
    return m * (segment * s - b)
  end
  indices = 1:(2^n - 1)
  accuracy_vector = [Float64(triangle(count_ones(i))) for i in indices]
  fitness_landscape = [Float64(triangle(count_ones(i))) - epsilon * count_ones(i) for i in indices]

  return accuracy_vector, fitness_landscape, n
end

function generate_test_synthetic(;n::Int = 31, epsilon = 0.0)
  if n != 31
    error("Step 6 asymmetric synthetic landscape is defined for n = 31 in the assignment test description.")
  end
  by_active_bits = UInt8[0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 6]

  num_elements = 2^n - 1
  
  fitness_landscape = Vector{UInt8}(undef, num_elements)

  for i in 1:num_elements
      fitness_landscape[i] = by_active_bits[count_ones(i) + 1]
  end

  return fitness_landscape, fitness_landscape, n
end