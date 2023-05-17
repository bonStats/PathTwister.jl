# Particle Structure #
mutable struct VectorParticle{d} <: AbstractParticle
    x::SVector{d, Float64}
    VectorParticle{d}() where d = new()
end
##

# Markov Kernel #
struct LinearGaussMarkovKernel{R<:Real} <: MarkovKernel
    A::AbstractMatrix{R}
    b::Vector{R}
    Σ::AbstractMatrix{R}
end

(M::LinearGaussMarkovKernel{R})(x::AbstractVector{R}) where {R<:Real} = MvNormal(M.A * x + M.b, M.Σ)
##

# Potential Structure #
struct MvNormalNoise <: LogPotentials
    obs::Vector{MvNormal}
end

MvNormalNoise(y::Vector{Vector{R}}, Σ) where {R<:Real} = MvNormalNoise([MvNormal(y[p], Σ) for p in 1:length(y)])

function (G::MvNormalNoise)(p::Int64, particle::AbstractParticle, ::Nothing)
    return logpdf(G.obs[p], value(particle))
end

PathTwister.untwist(G::MvNormalNoise) = G