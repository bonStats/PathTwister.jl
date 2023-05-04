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


# Particle Structure #
mutable struct TTVectorParticle{d} <: AbstractParticle # TT = Tempered Twist by β
    x::SVector{d, Float64}
    β₀::Float64 # current
    β₁::Float64 # next
    TTVectorParticle{d}() where d = new()
end
##

# Twisted Distribution

function (M::LinearGaussMarkovKernel{R})(p::TTVectorParticle, ψ::AbstractTwist) where {R<:Real}
    d = MvNormal(M.A * p.x + M.b, M.Σ)
    return RejectionSampler(d, ψ, p.β₁, MAXITER)
end

# Tempering Value Markov Kernel β

struct TemperAdaptSampler{T<:AbstractTwist} <: Sampleable{Univariate, Continuous}
    d::Sampleable{<:VariateForm,<:ValueSupport}
    ψ::T
    logα::Float64 # acceptance target
    Nₐ::Int64 # sample to estimate acceptance rate
end

function Base.rand(rng::AbstractRNG, s::TemperAdaptSampler{T}) where {T<:AbstractTwist}
        
    # choose β from trial draws
    logψx = s.ψ(rand(rng, s.d, s.Nₐ), :log)

    # define reach_acc_rate(b) > 0 if accept target is exceeded
    reach_acc_rate(b::Float64) = logsumexp(b .* logψx) - log(s.Nₐ) - s.logα
    if reach_acc_rate(1.0) > 0
        β = 1.0
    else
        β = find_zero(reach_acc_rate, (0,1))
    end

    return β
end

struct TemperTwist <: MarkovKernel
    logα::Float64 # acceptance target
    Nₐ::Int64 # sample to estimate acceptance rate
end

(Mβ::TemperTwist)(d::Sampleable, ψ::AbstractTwist) = TemperAdaptSampler(d, ψ, Mβ.logα, Mβ.Nₐ)


# function (Mβ::TemperTwist)(x::AbstractVector{R}, μ::AdaptiveRejection{S,T}, ψ::AbstractTwist) where {R<:Real,S<:Sampleable,T<:AbstractTwist}
#     return TemperAdaptSampler(μ, ψ, Mβ.logα, Mβ.Nₐ)
# end



struct AdaptiveTwistedMarkovChain{D<:Sampleable,K<:MarkovKernel,T<:AbstractTwist} <: AbstractMarkovChain
    μ::D
    M::K
    Mβ::TemperTwist
    n::Int64
    ψ::AbstractVector{T}
end

function Base.getindex(chain::AdaptiveTwistedMarkovChain{D,K,T}, i::Integer; base::Bool = false) where {D<:Sampleable,K<:MarkovKernel,T<:AbstractTwist}

    if base
        if i == 1
            return chain.μ
        else
            return (chain.M isa AbstractVector) ? chain.M[i-1] : chain.M
        end
    end

    if i == 1
        return (tt) -> AdaptiveRejection(chain.μ, chain.ψ[1], tt.logα, tt.Nₐ, MAXITER)
    elseif i <= length(chain)
        return (old) -> RejectionSampler(chain.M[i-1](old), ψ[i], old.β₁, MAXITER)
    else
        @error "Index $i not defined for chain."
    end
 end

function (chain::AdaptiveTwistedMarkovChain{D,K,T})(new::TTVectorParticle, rng, p::Int64, old::TTVectorParticle, ::Nothing) where {D<:Sampleable,K<:MarkovKernel,T<:AbstractTwist}
    # function changes! new::TTVectorParticle
    Mψx = (p == 1) ? chain[1](chain.Mβ)(rng) : chain[p](old) 
    # chain[p] == twisted mutation/distribution at p
    
    # mutate: x
    newx = Array{eltype(Mψx)}(undef, size(Mψx))
    rand!(rng, Mψx, newx)
    new.x = newx

    # mutate: current β
    new.β₀ = (p == 1) ? Mψx.β : old.β₁

    # mutate: next βx
    if p < length(chain)
        Mβ = chain.Mβ(chain[p+1, base = true](newx), chain.ψ[p+1])
        new.β₁ = rand(rng, Mβ)
    end

end


Mβ = TemperTwist(log(0.05), 5)
chainψ = AdaptiveTwistedMarkovChain(μ, M, Mβ, n, bestψ)

# test twisted chain...
p = TTVectorParticle{d}()
p.x = randn(2)
p.β₀ = p.β₁ = 0.01

oldp = deepcopy(p)

chainψ(p, Random.GLOBAL_RNG, 1, oldp, nothing)
p

chainψ(p, Random.GLOBAL_RNG, 2, oldp, nothing)
p

# Twisted Potential Structure
struct MCTwistedLogPotentials <: LogPotentials
    G::LogPotentials # original potentials
    M::AbstractMarkovChain # original chain
    ψ::AbstractVector{<:AbstractTwist}
    Nₘ::Int64
end

Gψ = MCTwistedLogPotentials(potential, chain, bestψ, Nα)

# needs to depend on rng in final version... (only serial for now)
function (Gψ::MCTwistedLogPotentials)(p::Int64, particle::TTVectorParticle, ::Nothing)
    
    logpot = logpdf(Gψ.G.obs[p], value(particle)) - Gψ.ψ[p](value(particle), :log)
    
    if p < length(Gψ.chain)
        newparticles = [typeof(particle)() for _ in 1:Gψ.Nₘ]
        Gψ.M.(newparticles, [Random.GLOBAL_RNG], [p+1], [particle], [nothing])
        logpot += logsumexp(particle.β .* Gψ.ψ[p+1](newparticles, :log)) - log(Gψ.Nₘ)
    end

    if p == 1
        newparticles = [typeof(particle)() for _ in 1:Gψ.Nₘ]
        Gψ.M.(newparticles, [Random.GLOBAL_RNG], [1], [particle], [nothing]) # doesn't use particle
        logpot += logsumexp(β .* Gψ.ψ[1](newparticles, :log)) - log(Gψ.Nₘ)
    end

    return logpot
end
##