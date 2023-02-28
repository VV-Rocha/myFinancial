using LinearAlgebra

function sinc(x::Vector{Float64}) ::Vector{Float64}
    return x
end

function mse(y_pred, y_expected) ::Float64
    return sum(sqrt.((y_pred.-y_expected).^2))
end

function ridge(x::Matrix{Float64}, y::Matrix{Float64}, lambda::Float64) ::Matrix{Float64}
    x_T::Matrix{Float64} = transpose(x)
    weights::Matrix{Float64} = inv(x_T*x + lambda*Matrix(I, size(x_T,1), size(x_T, 1))) * x_T * y
    return weights
end


N = 1000

noise = randn(Float64, N)

x_train = range(-10., 10., N) + noise

Y_train = sinc(x_train)

X_train = reshape(x_train, (size(x_train, 1), 1))
y_train = reshape(Y_train, (size(Y_train, 1), 1))

@time w = ridge(X_train, y_train, 0.0)

#print(size(X_train*w), "\n")
#print(size(y_train), "\n")
#print(mse(y_train, X_train*w), "\n")
