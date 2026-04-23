function generate_synthetic_fitness_landscape(;m::Int = 1, s::Int = 4, n::Int = 16, epsilon = 0.01)
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

  # Keep indexing compatible with parse_file/make_evaluate:
  # i = 1..(2^n - 1), where i is the decimal bit-mask representation.
  indices = 1:(2^n - 1)
  accuracy_vector = [Float64(triangle(count_ones(i))) for i in indices]
  fitness_landscape = [Float64(triangle(count_ones(i))) - epsilon * count_ones(i) for i in indices]

  return accuracy_vector, fitness_landscape, n
end

function generate_step6_asymmetric_synthetic_landscape()
  # From the test data description (index = number of active bits):
  # [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 6]
  by_active_bits = UInt8[0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 6]

  indices = 1:(2^31 - 1)
  accuracy_vector = [Int8(by_active_bits[count_ones(i) + 1]) for i in indices]
  return accuracy_vector, Nothing, 31
end