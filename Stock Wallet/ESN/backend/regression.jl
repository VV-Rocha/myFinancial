
module Regression

using LinearAlgebra

function ridge(x::Matrix{Float64}, y::Matrix{Float64}, lambda::Float64) ::Matrix{Float64}
    #print(size(x), " ", size(y), "\n")
    nfeatures, nsamples = size(x)
    x_T::Matrix{Float64} = transpose(x)

    weights::Matrix{Float64} = y*x_T*inv(x*x_T .+ lambda .* Matrix(I, nfeatures, nfeatures))
    #weights::Matrix{Float64} = inv(x_T*x .+ lambda .* Matrix(I, size(x,2), size(x, 2))) * x_T * y
    return weights
end

export ridge

end