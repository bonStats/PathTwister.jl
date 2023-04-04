
struct MarkovChain{D,K} <: AbstractMarkovChain # e.g. D = Distribution, K = MarkovKernel
    μ::D
    M::Vector{K}
end

Base.length(chain::MarkovChain{D,K}) where {D,K} = 1 + length(chain.M)

function Base.getindex(chain::MarkovChain{D,K}, i::Integer) where {D,K}
   @assert 1 <= i <= length(chain)
   i == 1 ? chain.μ : chain.M[i-1]
end

struct HomogeneousMarkovChain{D,K} <: AbstractMarkovChain # e.g. D = Distribution, K = MarkovKernel
    μ::D
    M::K
    n::Int64
end

Base.length(chain::HomogeneousMarkovChain{D,K}) where {D,K} = chain.n

function Base.getindex(chain::HomogeneousMarkovChain{D,K}, i::Integer) where {D,K}
   @assert 1 <= i <= length(chain)
   i == 1 ? chain.μ : chain.M
end

MarkovChain(μ::D, M::K, n::Int64) where {D,K} = HomogeneousMarkovChain(μ, M, n)

