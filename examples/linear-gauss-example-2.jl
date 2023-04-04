using SequentialMonteCarlo
using Distributions
using LinearAlgebra
using StaticArrays
using Random
using PathTwister

mutable struct VectorParticle{d} <: AbstractParticle
    x::SVector{d, Float64}
    VectorParticle{d}() where d = new()
end

struct LinearGaussMarkovKernel{R<:Real} <: MarkovKernel
    A::AbstractMatrix{R}
    b::Vector{R}
    Σ::AbstractMatrix{R}
end

(M::LinearGaussMarkovKernel{R})(x::AbstractVector{R}) where {R<:Real} = MvNormal(M.A * x + M.b, M.Σ)

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

noise = MvNormal(d,1.0)

chain = MarkovChain(μ, repeat([M], n-1))

# sim data
n = 10
latentx = repeat([rand(μ)], n)
y = repeat([latentx[1] + rand(noise)], n)
for p in 2:n
    latentx[p] = rand(M(latentx[p-1]))
    y[p] = latentx[p] + rand(noise)
end

# generalise this...
function ℓG(p::Int64, particle::AbstractParticle, ::Nothing)
    return logpdf(MvNormal(y[p], 1.0*I), value(particle))
end

model = SMCModel(chain, ℓG, n, VectorParticle{d}, Nothing)

smcio = SMCIO{model.particle, model.pScratch}(2^10, 10, 1, true)

smc!(model, smcio)


bestψ = [ExpQuadTwist{Float64}(d) for _ in 1:model.maxn]

lassocvtwist!(bestψ, smcio, model, 4, cvstrategy = 8)

bestψ[1].J

