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


mutable struct ExpQuadTwist{R<:Real} <: AbstractTwist
    h::AbstractVector{R}
    J::AbstractMatrix{R}
    # ExpQuadTwist{R}() where {R<:Real} = new()
    # ExpQuadTwist(h::AbstractVector{R}, J::AbstractMatrix{R}) where {R<:Real} = new{R}(h,J)
end

ExpQuadTwist{R}(d::Int64) where {R<:Real} = ExpQuadTwist(zeros(R,d),zeros(R,d,d))

function evolve!(ψ::ExpQuadTwist{R}, h::AbstractVector{R}, J::AbstractMatrix{R}) where {R<:Real}
    ψ.h = ψ.h + h
    ψ.J = ψ.J + J
end

function (ψ::ExpQuadTwist{R})(x::AbstractVector{R}, outscale::Symbol) where {R<:Real}
    d = MvNormalCanon(m.h, m.J)
    ℓpdf = logpdf(d, x) - logpdf(d, mode(d)) # maximum: log(1) = 0
    if outscale == :log
        return ℓpdf
    else
        @warn "Returning on standard scale, logscale = $ℓpdf"
        exp(ℓpdf)
    end
end

function (ψ::ExpQuadTwist{R})(particles::Vector{<:P}, outscale::Symbol) where {R<:Real, P<:AbstractParticle}
    d = MvNormalCanon(ψ.h, ψ.J)
    ℓpdf = logpdf(d, value(particles)) .- logpdf(d, mode(d)) # maximum: log(1) = 0
    if outscale == :log
        return ℓpdf
    else
        @warn "Returning on standard scale, e.g. logscale[1] = $ℓpdf"
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

using GLMNet

function buildresponse(particles::Vector{<:P}, M!::Function, ℓG::Function, p::Int64, ψ::Union{T, Nothing} = nothing, rng = Random.GLOBAL_RNG) where {P<:AbstractParticle,T<:AbstractTwist}
    ℓresp = ℓG.([p], particles, nothing) #ℓGp
    if !isnothing(ψ) # when p == n
        newparticles = [model.particle() for i in 1:length(particles)]
        M!.(newparticles, [rng], [p+1], particles, [nothing])
        ℓresp += ψ(newparticles, :log)
    end
    return ℓresp
end

function buildpredictors(particles::Vector{<:P}) where {P<:AbstractParticle}
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

function learntwistlassocv(X::AbstractMatrix{R}, y::AbstractVector{R}, parids::Dict{Symbol, UnitRange{Int64}}, folds::Union{AbstractVector{Int64}, Int64}, quadupper::AbstractVector{R}) where {R<:Real}
    quadconstraint = repeat([-Inf; Inf], 1, size(X, 2))
    quadconstraint[:,parids[:quad]] .= [repeat([-Inf], length(parids[:quad])) quadupper]'
    if isa(folds, AbstractVector)
        return glmnetcv(X, y, folds = folds, constraints = quadconstraint)
    else isa(folds, Integer)
        return glmnetcv(X, y, nfolds = folds, constraints = quadconstraint)
    end
end

reps = 5
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
res = learntwistlassocv(predictorsX, responsey, ids, foldid, zeros(d))


# Lasso Regression -> MvNormalCanon form
# ignoring constants...
# ∑aᵢᵢxᵢ² + ∑aᵢⱼxᵢxⱼ1(i<j) + ∑bᵢxᵢ = -0.5(x-μ)ᵀP(x-μ) = -0.5xᵀPx + xᵀPμ = -0.5xᵀJx + xᵀh
# e.g. in two dimensions:
# |x₁|ᵀ * |a b| * |x₁|   | ax₁ + bx₂|ᵀ |x₁|   
# |x₂|    |b c|   |x₂| = | bx₁ + cx₂|  |x₂| = ax₁² + 2bx₁x₂ + cx₂²
function pathcv_hJ(pathcv::GLMNetCrossValidation, parids::Dict{Symbol, UnitRange{Int64}})
    β = coef(pathcv)
    
    h = β[parids[:linear]]
    
    d = length(parids[:linear])
    J = zeros(d,d)
    J[diagind(J)] = -2 .* β[parids[:quad]]
    for (i, cc) in uppertriiter(d)
        J[cc] = J[reverse(cc)] = -β[parids[:cross][i]]
    end

    #h, J: MvNormalCanon
    h, J
end

# maximum adjustment for ψ in terms of constraints
# on quadratic diagonal terms from lassocv
maxlassoadjust(ψ::ExpQuadTwist{R}, ϵ::Float64) where {R<:Real} = 0.5*diag(ψ.J) .- ϵ

# function ExpQuadTwist(pathcv::GLMNetCrossValidation, parids::Dict{Symbol, UnitRange{Int64}}, zeroquadϵ::Float64 = 1e-4)
#     h, J = pathcv_hJ(pathcv, parids)
#     # correct any quadratic components = 0 if zeroquadϵ ≢ 0
#     if !iszero(zeroquadϵ)
#         J[diagind(J)] = map(x -> iszero(x) ? abs(zeroquadϵ) : x, J[diagind(J)])
#     end
#     return ExpQuadTwist(h, J)
# end

ψ = ExpQuadTwist(res, ids)
MvNormalCanon(ψ.h, ψ.J)

# p < n
p = 9
particles = smcio.allZetas[9]
X_single, ids = buildpredictors(particles)

responsey = vcat([buildresponse(particles, M!, ℓG, p, ψ) for i in 1:reps]...)
predictorsX = repeat(X_single, reps) # more space efficient version?
foldid = vcat([i*ones(Int64, size(X_single,1)) for i in 1:reps]...)
res = learntwistlassocv(predictorsX, responsey, ids, foldid, zeros(d))

ψ2 = ExpQuadTwist(res, ids)
MvNormalCanon(ψ2.h, ψ2.J)

function lassocvtwist!(ψ::Vector{ExpQuadTwist{R}}, smcio::SMCIO{P, S}, model::SMCModel, MCreps::Int64, cvstrategy::Union{Symbol, Int64} = :byreps, quadϵ::Float64 = 1e-02) where {R<:Real, P<:AbstractParticle, S}
    # TD: how to check if smcio has been run?
    # TD: Need scratch for extra particles
    if !smcio.fullOutput
        @error "smcio must have full particle trajectory"
    end

    # setup cv strategy
    if (MCreps > 1) & (cvstrategy == :byreps)
        # folds = id vector which fold an observation belongs to
        folds = vcat([i*ones(Int64, smcio.N) for i in 1:MCreps]...)
    elseif (MCreps > 0) & isa(cvstrategy, Integer)
        # nfolds = how many folds to use
        folds = cvstrategy
    else
        @error "Incompatible MCreps and CV strategy"
    end

    for p in model.maxn:-1:1
        # single design matrix
        X_single, parids = buildpredictors(smcio.allZetas[p])

        if p == model.maxn
            # extra particles for same number of observations over p
            if MCreps > 1
                padparticles = [[model.particle() for _ in 1:smcio.N] for _ in 2:MCreps]
                pushfirst!(padparticles, smcio.allZetas[model.maxn])
                for t in 2:MCreps
                    model.M!.(padparticles[t], [Random.GLOBAL_RNG], [model.maxn], smcio.allZetas[model.maxn-1], [nothing]) #rng
                end
            else 
                padparticles = smcio.allZetas[model.maxn]
            end

            y = vcat([buildresponse(padparticles[i], model.M!, model.lG, model.maxn) for i in 1:MCreps]...)
            X = vcat(X_single, [buildpredictors(padparticles[i], nothing) for i in 2:MCreps]...)
            
        else # p < model.maxn
            y = vcat([buildresponse(smcio.allZetas[p], model.M!, model.lG, p, ψ[p+1]) for i in 1:MCreps]...)
            X = repeat(X_single, MCreps) # space efficient version?
        end

        h, J = pathcv_hJ(learntwistlassocv(X, y, parids, folds, maxlassoadjust(ψ[p], abs(quadϵ))), parids)

        evolve!(ψ[p], h, J)

    end
end


bestψ = [ExpQuadTwist{Float64}(d) for _ in 1:model.maxn]

lassocvtwist!(bestψ, smcio, model, 5)
