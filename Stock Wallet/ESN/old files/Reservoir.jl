
include("./sparse_matrices.jl")
#include("../Data reader/data_reader.jl")
using .sparse_matrices
#using .data_reader


using SparseArrays
using Statistics
using LinearAlgebra

module ESN

struct Reservoir
    # parameters
    nodes::Int64
    leak::Float64
    ndrop::Int64
    noise_factor::Float64

    ## primary
    w_in::Matrix{Float64}
    w_reservoir::Matrix{Float64}

    #calculated through regression
    w_out::Matrix{Float64}

    ## optional
end

function ridge(x::Matrix{Float64}, y::Matrix{Float64}, lambda::Float64) ::Matrix{Float64}
    x_T::Matrix{Float64} = transpose(x)
    weights::Matrix{Float64} = inv(x_T*x .+ lambda*Matrix(I, size(x,2), size(x, 2))) * x_T * y
    return weights
end

function evolve_system(X_train::Array{Float64}, nodes::Int64, w_in::Matrix{Float64}, w_reservoir::Matrix{Float64}, leak::Float64, ndrop::Int64, noise_factor::Float64) ::Array{Float64, 3}
    nsamples, _, integration_time = size(X_train)
    
    p_states ::Array{Float64} = zeros(Float64, (nsamples, nodes))

    states_history ::Array{Float64, 3} = zeros((nsamples, nodes, integration_time-ndrop))
    Threads.@threads for t::Int64 in range(1, integration_time, step=1)
        noise = rand(size(X_train, 1), nodes)
        states = tanh.(X_train[:,:,t]*w_in .+ p_states*w_reservoir .+ noise .* noise_factor)
        states = (1. -leak).*p_states .+ states
        if t > ndrop
            states_history[:,:,t-ndrop] = states
        end
    end
    return states_history
end

function predict(X, reservoir)
    states_history = evolve_system(X, reservoir.nodes, reservoir.w_in, reservoir.w_reservoir, reservoir.leak, reservoir.ndrop, reservoir.noise_factor)
    #states_history = reshape(mean(states_history, dims=3), (size(states_history, 1), size(states_history, 2), 1))
    print(">> ",states_history[1,1,1], "\n")
    likelihoods = reshape(states_history, (size(states_history,1), size(states_history,2)*size(states_history,3))) * reservoir.w_out
    return likelihoods
end

export Reservoir
export ridge
export evolve_system
export predict

end

function maxindex(arr)
    r = zeros(size(arr,1))
    for i = 1:size(arr,1)
        r[i] = findall(arr[i, :] .== findmax(arr[i, :])[1])[1]
    end
    #r = [findall(arr[:,i].==max(arr[i]))[1] for i = 1:size(arr,1)]
    return r
end

function generate_reservoir(X_train::Array{Float64, 3},
                            y_train::Matrix{Float64},
                            nodes::Int64,
                            reservoir_density::Float64,
                            leak::Float64,
                            ndrop::Int64,
                            lambda::Float64,
                            noise_factor::Float64,
                            in_out::Bool=false)
    nclass = size(unique(y_train), 2)
    axis = size(X_train, 2)

    # generate random matrices (w_in, w_in_out, w_reservoir)
    w_in ::Matrix{Float64} = sparse_random_matrix(size(X_train, 2), nodes, 1.) .* 1.
    w_reservoir ::Matrix{Float64} = sparse_random_matrix(nodes, nodes, reservoir_density)
    #print(eigvals(w_reservoir), "\n")
    w_reservoir ./= findmax(abs.(eigvals(w_reservoir)))[1] /0.99
    # evolve states
    states_history ::Array{Float64} = evolve_system(X_train, nodes, w_in, w_reservoir, leak, ndrop, noise_factor)

    # determine w_out, i.e. train the model
    w_out ::Matrix{Float64} = ridge(reshape(states_history, (size(states_history,1), size(states_history,2)*size(states_history,3))), y_train, lambda)
    print("> ", states_history[1, 1, 1], "\n")
    reservoir = Reservoir(nodes, leak, ndrop, noise_factor, w_in, w_reservoir, w_out)
    return reservoir
end



function repeat(arr, nrepetitions) ::Array{String}
    repeated_arr::Array{String} = []
    for i in arr
        for j in 1:nrepetitions
            repeated_arr = vcat(repeated_arr, i)
        end
    end
    return reshape(repeated_arr, (size(repeated_arr,1), size(repeated_arr, 2)))
end

function setindex(arr, value, index)
    arr[index] = value
    return arr
end

function onehot(arr) ::Matrix{Float64}
    entities = unique(arr)
    onehot = zeros(size(entities, 1), size(arr, 1))
    for i = 1:size(arr,1)
        onehot[findall(arr[i].==entities)[1], i] = 1
    end
    return transpose(onehot)
end

"""
print("Loading data:", "\n")

dir = "../../Data/1st article data/Synthetic particles/"
npoints = 1000
classes = ["3umPMMA", "8umPMMA"]
naxis = 2
data = load_data(dir, npoints, classes, naxis)

nparticles = size(data,1)/size(classes, 1)
data = data[:,1:10,:,:]

targets = repeat(classes, nparticles*size(data, 2))

targets = onehot(targets)  # one-hot enconding

data = reshape(data, (size(data,1)*size(data,2), size(data,3), size(data,4)))

#data = data[:,:,1:1000]

data[:, 1, :] ./= findmax(data[:,1,:])[1]
data[:, 2, :] ./= findmax(data[:,2,:])[1]

print("Done!", "\n")

reservoir = generate_reservoir(data, targets, 10, 0.1, 0.8, 10, 10^-4, 0.0)

#print(reservoir.nodes, " ", reservoir.leak, " ", reservoir.ndrop, " ", reservoir.leak, " ", reservoir.noise_factor, "\n")

y = [findall(targets[i, :].==findmax(targets[i,:])[1])[1] for i = 1:size(targets, 1)]

#print(y, "\n")

likelihoods = predict(data, reservoir)
#print(likelihoods, "\n")
prediction = maxindex(likelihoods)
#print(prediction, "\n")
print(size(findall(prediction.==y), 1)/size(data, 1), "\n")
"""