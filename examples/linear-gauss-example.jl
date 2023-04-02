using SequentialMonteCarlo
using Distributions
using Random
using LinearAlgebra
using StaticArrays

# uppertri iterator
uppertriiter(d::Int64) = enumerate(Iterators.filter(c -> c[1] < c[2], CartesianIndices((1:d,1:d))))
Base.reverse(cc::CartesianIndex) = CartesianIndex(cc[2], cc[1])
abstract type AbstractParticle end
abstract type AbstractTwist end

# fallback...
value(particle::AbstractParticle) = particle.x
value(particles::Vector{P}) where {P<:AbstractParticle} = reduce(hcat, value.(particles))

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


struct ExpQuadTwist{R<:Real} <: AbstractTwist
    h::AbstractVector{R}
    J::AbstractMatrix{R}
end

function (ψ::ExpQuadTwist{R})(x::AbstractVector{R}, logscale::Bool = true) where {R<:Real}
    d = MvNormalCanon(m.h, m.J)
    ℓpdf = logpdf(d, x) - logpdf(d, mode(d)) # maximum: log(1) = 0
    if logscale
        return ℓpdf
    else
        exp(ℓpdf)
    end
end

function (ψ::ExpQuadTwist{R})(particles::Vector{<:P}, logscale::Bool = true) where {R<:Real, P<:AbstractParticle}
    d = MvNormalCanon(m.h, m.J)
    ℓpdf = logpdf(d, value(particles)) .- logpdf(d, mode(d)) # maximum: log(1) = 0
    if logscale
        return ℓpdf
    else
        exp.(ℓpdf)
    end
end

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

particles = smcio.allZetas[10]

using GLMNet

function buildresponse(particles::Vector{<:P}, M!::Function, ℓG::Function, p::Int64, ψ::Union{AbstractTwist, Nothing} = nothing, rng = Random.GLOBAL_RNG) where {P<:AbstractParticle}
    ℓresp = ℓG.([p], particles, nothing) #ℓGp
    if !isnothing(ψ) # when p == n
        newparticles = typeof(particles)(undef, length(particles))
        M!.(newparticles, [rng], [p+1], particles, [nothing])
        ℓresp += ψ.(newparticles)
    end
    return ℓresp
end

function buildpredictors(particles::Vector{<:P}, ids::Bool = true) where {P<:AbstractParticle}
    d = length(value(particles[1])) # original predictors
    r = 2*d+binomial(d,2) # regression size (ignore intercept)
    X = ones(length(particles),r)
    # populate
    quadid = 1:d
    linearid = quadid .+ d
    crossid = (max(maximum(linearid),maximum(quadid))+1):r
    X[:,linearid] = value(particles)'
    X[:,quadid] = X[:,linearid] .^2
    for (i, cc) in uppertriiter(d)
        X[:,crossid[i]] = X[:,linearid[cc[1]]] .* X[:,linearid[cc[2]]]
    end
    X, Dict(:linear => linearid, :quad => quadid, :cross => crossid)
end

# don't return ids
function buildpredictors(particles::Vector{<:P}, ::Nothing) where {P<:AbstractParticle}
    X, _ = buildpredictors(particles)
    return X
end


reps = 10
# p = n
p = 10
padparticles = [[model.particle() for i in 1:length(particles)] for t in 2:reps]
pushfirst!(padparticles, smcio.allZetas[10])
for t in 2:reps
    M!.(padparticles[t], [Random.GLOBAL_RNG], [p], smcio.allZetas[9], [nothing]) #rng
end
X_single, ids = buildpredictors(padparticles[1])

responsey = vcat([buildresponse(padparticles[i], M!, ℓG, p) for i in 1:reps]...)
predictorsX = vcat(X_single, [buildpredictors(padparticles[i], nothing) for i in 2:reps]...)
foldid = vcat([i*ones(Int64, size(X_single,1)) for i in 1:reps]...)
quadconstraints = repeat([-Inf; Inf], 1, size(X_single, 2))
quadconstraints[:,ids[:quad]] .= repeat([-Inf; 0.0], 1, length(ids[:quad]))
res = glmnetcv(predictorsX, responsey, folds = foldid, constraints = quadconstraints)
cvcoefres = coef(res)

cvcoefres[ids[:quad]]
cvcoefres[ids[:linear]]
cvcoefres[ids[:cross]]

# Regression to MvNormalCanon form
# ignoring constants...
# ∑aᵢᵢxᵢ² + ∑aᵢⱼxᵢxⱼ1(i<j) + ∑bᵢxᵢ = -0.5(x-μ)ᵀP(x-μ) = -0.5xᵀPx + xᵀPμ = -0.5xᵀJx + xᵀh
# |x₁|ᵀ * |a b| * |x₁|   | ax₁ + bx₂|ᵀ |x₁|   
# |x₂|    |b c|   |x₂| = | bx₁ + cx₂|  |x₂| = ax₁² + 2bx₁x₂ + cx₂²

J = zeros(2,2)
J[diagind(J)] = -2 .* cvcoefres[ids[:quad]]
for (i, cc) in uppertriiter(d)
    J[cc] = J[reverse(cc)] = -cvcoefres[ids[:cross][i]]
end
eq = ExpQuadTwist(cvcoefres[ids[:linear]], J)

MvNormalCanon(eq.h, eq.J)

# p < n
p = 9
particles = smcio.allZetas[9]
X_single, ids = buildpredictors(particles)

responsey = vcat([buildresponse(particles, M!, ℓG, p, ψ) for i in 1:reps]...)
predictorsX = repeat(X_single, reps) # more space efficient version?
foldid = vcat([i*ones(Int64, size(X_single,1)) for i in 1:reps]...)
quadconstraints = repeat([-Inf; Inf], 1, size(X_single, 2))
quadconstraints[:,ids[:quad]] .= repeat([-Inf; 0.0], 1, length(ids[:quad]))
res = glmnetcv(predictorsX, responsey, folds = foldid, constraints = quadconstraints)
cvcoefres = coef(res)