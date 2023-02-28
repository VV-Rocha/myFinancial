import Pkg
#Pkg.add("StatsBase") 
#Pkg.add("Distributions")

using SparseArrays
using InteractiveUtils

module Sparse

using SparseArrays
using StatsBase
using Distributions
using Random

export sparse_random_matrix

function remove!(a, item)
    deleteat!(a, findall(x->x==item, a))
end

function counter(ncols::Int64, vector::Array{Int64, 1}) ::Array{Int64, 1} #::Tuple{Vector{Int64}, Vector{Int64}}
    """
    Counts the occurrences of a sorted vector of intengers

    Parameters:
        -> vector::Vector{Int64} - vector containing a set of indices
    """
    #@time begin
    counter::Array{Int64, 1} = fit(Histogram, vector, 1:ncols+1).weights
    #end

    # clean counter from zeros
    remove!(counter, 0)

    return counter #entities, counter
end

function get_rows(nrows::Int64, col_count::Array{Int64, 1}) ::Array{Int64, 1}
    row_indices::Array{Int64, 1} = zeros(sum(col_count))
    cs::Array{Int64, 1} = cumsum(col_count)
    cs = vcat([0], cs)
    for c = range(1, size(col_count, 1), step=1)
        row_indices[cs[c]+1:cs[c+1]] = sample(1:nrows, col_count[c], replace=false)
    end

    return row_indices
end

function fill_matrix(nrows::Int64, ncols::Int64, col_indices::Array{Int64, 1}, col_count::Array{Int64, 1}) ::Matrix{Float64}
    #@time begin
    #row_indices::Array{Int64, 1} = get_rows(nrows, col_count)
    #end
    #@time begin
    matrix::Matrix{Float64} = zeros(Float64, nrows, ncols)
    for col = 1:size(col_indices, 1)
        matrix[sample(1:nrows, col_count[col], replace=false), col_indices[col]] = randn(Float64, col_count[col])
    end
    #matrix = sparse(row_indices, col_indices, randn(Float64, size(col_indices,1)))
    #end
    return matrix
end

function get_column_indices(nrows::Int64, ncols::Int64, ntotal_elements::Int64) ::Tuple{Array{Int64, 1}, Array{Int64, 1}}
    indices = zeros(Int64, nrows, ncols)
    for i::Int64 in range(1, ncols, step=1)
        indices[:, i] .= i    
    end

    #@time begin
    col_indices::Array{Int64, 1} = sample(indices, ntotal_elements, replace=false, ordered=true)
    #end
    #@time begin
    col_count ::Array{Int64, 1} = counter(ncols, col_indices)
    #end
    return col_indices, col_count
end

function sparse_random_matrix(nrows::Int64,
                              ncols::Int64,
                              density::Float64) #::Matrix{Float64}
    #@time begin
    ntotal_elements ::Int64 = floor(Int64, (nrows*ncols*density))
    #end
    
    #@time begin
    # generate random column indices (and sort them)
    col_indices ::Array{Int64, 1}, col_count::Array{Int64, 1} = get_column_indices(nrows, ncols, ntotal_elements)
    #end
    
    #@time begin
    matrix  = fill_matrix(nrows, ncols, unique(col_indices), col_count)
    #end
    return matrix
end

end

#function asd(a::Int64, density::Float64) ::Matrix{Float64}
#    w_in::Matrix{Float64} = Matrix(sprand(a, a, density))
#    return w_in
#end

#asd(10, 1.)
#@time asd(1000, 1.)

#print(35.0 - 100.0*size(findall(w_in.!=0.),1)/(10000*10000), "\n")

#@time sparse_random_matrix(1000, 1000, 1.)

#@code_warntype sparse_random_matrix(10, 10, 1.)
#@trace(sparse_random_matrix(10, 10, 1.), maxdepth=3, modules=[Main])
#print(sparse_random_matrix(10,10, 1.0))

#
#sparse_random_matrix(10, 10, 1.)
#@time sparse_random_matrix(1000, 1000, 1.)

#print(20.0 - 100.0*size(findall(sparse_random_matrix(10, 10, 0.2).!=0.),1)/(1000*1000), "\n")

#print(sparse_random_matrix(10, 10, 1.))
