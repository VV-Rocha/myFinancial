

module ESN

using LinearAlgebra

include("./backend/sparse_matrices.jl")
using .Sparse

include("./backend/regression.jl")
using .Regression

struct Reservoir
    """
    Core of the Reservoir. This structure records the internal weights
    and, through training, the output weights which allows this information
    to be used for classifying or regression.
    """
    # parameters
    nodes::Int64
    radius::Float64
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

function generate_reservoir(X_train::Array{Float64, 3},
                            y_train::Matrix{Float64};
                            nodes::Int64=100,
                            radius::Float64=0.99,
                            reservoir_density::Float64 = 0.1,
                            leak::Float64 = 0.2,
                            ndrop::Int64 = 10,
                            lambda::Float64 = 0.,
                            noise_factor::Float64 = 1.,
                            in_out::Bool=false)

    """
    This function creates a reservoir structure containing the internal weights
    aswell as the output weight matrix.
    
    Parameter:
        -> X_train - [features, samples, time series]
        -> y_train - [output_features, samples]
    """
    
    nclass = size(unique(y_train), 2)
    axis = size(X_train, 2)

    nfeatures, nsamples, integration_time = size(X_train)

    # generate random matrices (w_in, w_in_out, w_reservoir)
    w_in ::Matrix{Float64} = Sparse.sparse_random_matrix(nodes, nfeatures, 1.) .* 1.
    w_reservoir ::Matrix{Float64} = Sparse.sparse_random_matrix(nodes, nodes, reservoir_density)
    #print(eigvals(w_reservoir), "\n")
    w_reservoir ./= findmax(abs.(eigvals(w_reservoir)))[1] /radius
    # evolve states
    states_history ::Array{Float64} = evolve_system(X_train, nodes, w_in, w_reservoir, leak, ndrop, noise_factor)

    # determine w_out, i.e. train the model
    #states_history = reshape(states_history, (nodes*(integration_time-ndrop), nsamples))
    #states_history = reshape(states_history[:,:,end], (nodes, nsamples))
    states_history = reshape(states_history[:,:,:], (nodes, nsamples*(integration_time-ndrop)))

    w_out ::Matrix{Float64} = Regression.ridge(states_history, y_train, lambda)

    reservoir = Reservoir(nodes, radius, leak, ndrop, noise_factor, w_in, w_reservoir, w_out)
    return reservoir
end

function evolve_system(X_train::Array{Float64}, nodes::Int64, w_in::Matrix{Float64}, w_reservoir::Matrix{Float64}, leak::Float64, ndrop::Int64, noise_factor::Float64) ::Array{Float64, 3}
    nfeatures, nsamples, integration_time = size(X_train)
    
    p_states ::Array{Float64} = zeros(Float64, (nodes, nsamples))

    states_history ::Array{Float64, 3} = zeros((nodes, nsamples, integration_time-ndrop))
    for t::Int64 in range(1, integration_time, step=1)
        noise = rand(nodes, nsamples) .* noise_factor
        states = tanh.(w_in*X_train[:,:,t] .+ w_reservoir*p_states .+ noise)
        states = (1. -leak).*p_states .+ states
        if t > ndrop
            states_history[:,:,t-ndrop] = states
        end
    end
    return states_history
end

function predict(X, reservoir)
    nfeatures, nsamples, integration_time = size(X)

    states_history = evolve_system(X,
                reservoir.nodes,
                reservoir.w_in,
                reservoir.w_reservoir,
                reservoir.leak,
                reservoir.ndrop,
                reservoir.noise_factor)


    #states_history = reshape(states_history, (reservoir.nodes*(integration_time-reservoir.ndrop), nsamples))
    #states_history = reshape(states_history[:, :, end], (reservoir.nodes, nsamples))
    states_history = reshape(states_history, (reservoir.nodes, nsamples*(integration_time-reservoir.ndrop)))

    return reservoir.w_out*states_history
end

export Reservoir
export generate_reservoir
export evolve_system
end