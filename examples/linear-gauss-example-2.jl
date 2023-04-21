using SequentialMonteCarlo
using Distributions
using LinearAlgebra
using StaticArrays
using Random
using PathTwister
using Roots
import StatsFuns: logsumexp

const MAXITER = 10^5

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
##

# setup problem
n = 10
d = 2
μ = MvNormal(d, 1.)

A = zeros(d,d)
A[diagind(A)] .= 0.5
A[diagind(A,-1)] .= 0.1
A[diagind(A,1)] .= 0.1

b = zeros(d)
Σ = Matrix(1.0*I, d, d)
M = LinearGaussMarkovKernel(A,b,Σ)

#chain = MarkovChain(μ, repeat([M], n-1))
chain = MarkovChain(μ, M, n)

# setup: sim data
noise = MvNormal(d, 1.0)
latentx = repeat([rand(μ)], n)
y = repeat([latentx[1] + rand(noise)], n)
for p in 2:n
    latentx[p] = rand(M(latentx[p-1]))
    y[p] = latentx[p] + rand(noise)
end

potential = MvNormalNoise(y, 1.0*I)

model = SMCModel(chain, potential, n, VectorParticle{d}, Nothing)

smcio = SMCIO{model.particle, model.pScratch}(2^10, 10, 1, true)

smc!(model, smcio)


bestψ = [ExpQuadTwist{Float64}(d) for _ in 1:model.maxn]

lassocvtwist!(bestψ, smcio, model, 4, cvstrategy = 8)

bestψ[1].J

# locally twisted SMC


# # Particle Structure #
# mutable struct TTVectorParticle{d} <: AbstractParticle # TT = Tempered Twist by β
#     x::SVector{d, Float64}
#     β::Float64
#     VectorParticle{d}() where d = new()
# end
# ##

# Twisted Distribution

ℓα = log(0.05)
Nα = 5

μψ = AdaptiveRejection(μ, bestψ[1], ℓα, Nα, MAXITER)


# Twisted Markov Kernel
struct TwistedLinearGaussMarkovKernel{R<:Real} <: MarkovKernel
    M::LinearGaussMarkovKernel{R}
    ψ::AbstractTwist
    logα::Float64 # acceptance target
    Nₐ::Int64 # sample to estimate acceptance rate
end

# vectorised constructor
TwistedLinearGaussMarkovKernel(
    M::LinearGaussMarkovKernel{R}, 
    ψ::AbstractVector{T}, 
    logα::Float64,  Nₐ::Int64) where {R<:Real,T<:AbstractTwist} = TwistedLinearGaussMarkovKernel.([M], ψ, [logα], [Nₐ])

function (M::TwistedLinearGaussMarkovKernel{R})(x::AbstractVector{R}) where {R<:Real} 
    # choose β from trial draws
    d = MvNormal(M.A * x + M.b, M.Σ)
    logψx = M.ψ.(rand(d, M.Nₐ), :log)

    # define reach_acc_rate(b) > 0 if accept target is exceeded
    reach_acc_rate(b::Float64) = logsumexp(b .* logψx) - log(M.Nₐ) - M.logα
    if reach_acc_rate(1.0) > 0
        β = 1.0
    else
        β = find_zero(reach_acc_rate, (0,1))
    end

    return RejectionSampler(d, M.ψ, β, MAXITER)

end

Mψ = TwistedLinearGaussMarkovKernel(M, bestψ[2:end], ℓα, Nα)

chainψ = MarkovChain(μψ, Mψ)



μψ()