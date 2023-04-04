using SequentialMonteCarlo
using Distributions
using LinearAlgebra
using StaticArrays
using Random
using PathTwister
# check works: this script works?
# try using Mutation/Potential that is functional structure

struct LinearGaussMarkovKernel{R<:Real}
    A::AbstractMatrix{R}
    b::Vector{R}
    Σ::AbstractMatrix{R}
end

function (M::LinearGaussMarkovKernel{R})(rng, x::AbstractVector{R}) where {R<:Real}
    μ = M.A * x + M.b
    return rand(rng, MvNormal(μ, M.Σ))
end
(M::LinearGaussMarkovKernel{R})(x::AbstractVector{R}) where {R<:Real} = M(Random.GLOBAL_RNG, x)


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

# NOTE: Add macro to do this...
function M!(new::AbstractParticle, rng, p::Int64, old::AbstractParticle, ::Nothing)
    if p == 1
      new.x = rand(rng, μ)
    else
      new.x = M(rng, old.x)
    end
end

mutable struct VectorParticle{d} <: AbstractParticle
    x::SVector{d, Float64}
    VectorParticle{d}() where d = new()
end

# sim data
n = 10
latentx = repeat([rand(μ)], n)
y = repeat([latentx[1] + rand(noise)], n)
for p in 2:n
    latentx[p] = M(latentx[p-1])
    y[p] = latentx[p] + rand(noise)
end

function ℓG(p::Int64, particle::AbstractParticle, ::Nothing)
    return logpdf(MvNormal(y[p], 1.0*I), value(particle))
end

model = SMCModel(M!, ℓG, n, VectorParticle{d}, Nothing)

smcio = SMCIO{model.particle, model.pScratch}(2^10, 10, 1, true)

smc!(model, smcio)


# reps = 5
# # p = n
# p = 10
# padparticles = [[model.particle() for i in 1:length(particles)] for t in 2:reps]
# pushfirst!(padparticles, smcio.allZetas[10])
# for t in 2:reps
#     M!.(padparticles[t], [Random.GLOBAL_RNG], [p], smcio.allZetas[9], [nothing]) #rng
# end
# X_single, ids = buildpredictors(padparticles[1])

# responsey = vcat([buildresponse(padparticles[i], M!, ℓG, p) for i in 1:reps]...)
# predictorsX = vcat(X_single, [buildpredictors(padparticles[i], nothing) for i in 2:reps]...)
# foldid = vcat([i*ones(Int64, size(X_single,1)) for i in 1:reps]...)
# res = learntwistlassocv(predictorsX, responsey, ids, foldid, zeros(d))


# function ExpQuadTwist(pathcv::GLMNetCrossValidation, parids::Dict{Symbol, UnitRange{Int64}}, zeroquadϵ::Float64 = 1e-4)
#     h, J = pathcv_hJ(pathcv, parids)
#     # correct any quadratic components = 0 if zeroquadϵ ≢ 0
#     if !iszero(zeroquadϵ)
#         J[diagind(J)] = map(x -> iszero(x) ? abs(zeroquadϵ) : x, J[diagind(J)])
#     end
#     return ExpQuadTwist(h, J)
# end

# ψ = ExpQuadTwist(res, ids)
# MvNormalCanon(ψ.h, ψ.J)

# # p < n
# p = 9
# particles = smcio.allZetas[9]
# X_single, ids = buildpredictors(particles)

# responsey = vcat([buildresponse(particles, M!, ℓG, p, ψ) for i in 1:reps]...)
# predictorsX = repeat(X_single, reps) # more space efficient version?
# foldid = vcat([i*ones(Int64, size(X_single,1)) for i in 1:reps]...)
# res = learntwistlassocv(predictorsX, responsey, ids, foldid, zeros(d))

# ψ2 = ExpQuadTwist(res, ids)
# MvNormalCanon(ψ2.h, ψ2.J)


bestψ = [ExpQuadTwist{Float64}(d) for _ in 1:model.maxn]

lassocvtwist!(bestψ, smcio, model, 4, cvstrategy = 8)

bestψ[1].J

