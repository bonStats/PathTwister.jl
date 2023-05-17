
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


struct AdaptiveTwistedMarkovChain{D<:Sampleable,K<:MarkovKernel,T<:AbstractTwist} <: AbstractMarkovChain
    μ::D
    M::K
    Mβ::TemperTwist
    n::Int64
    ψ::AbstractVector{T}
end

PathTwister.untwist(chain::AdaptiveTwistedMarkovChain) = MarkovChain(chain.μ, chain.M, chain.n)

function Base.getindex(chain::AdaptiveTwistedMarkovChain{D,K,T}, i::Integer; base::Bool = false) where {D<:Sampleable,K<:MarkovKernel,T<:AbstractTwist}

    if i == 1

        if base
            return chain.μ
        else 
            return (tt) -> AdaptiveRejection(chain.μ, chain.ψ[1], tt.logα, tt.Nₐ, MAXITER)
        end

    elseif i <= length(chain)

        M = (chain.M isa AbstractVector) ? chain.M[i-1] : chain.M

        if base
            return M
        else
            return (old) -> RejectionSampler(M(old), chain.ψ[i], old.β₁, MAXITER)
        end

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

# Twisted Potential Structure
struct MCTwistedLogPotentials <: LogPotentials
    G::LogPotentials # original potentials
    M::AbstractMarkovChain # original chain
    ψ::AbstractVector{<:AbstractTwist}
    Nₘ::Int64
end

PathTwister.untwist(Gψ::MCTwistedLogPotentials) = Gψ.G

# needs to depend on rng in final version... (only serial for now)
function (Gψ::MCTwistedLogPotentials)(p::Int64, particle::TTVectorParticle, ::Nothing)
    
    logpot = logpdf(Gψ.G.obs[p], value(particle)) - (particle.β₀ * Gψ.ψ[p](value(particle), :log))
    
    if p == length(Gψ.M)
        return logpot
    end
    
    newparticles = [typeof(particle)() for _ in 1:Gψ.Nₘ]
    
    if p < length(Gψ.M)
        Gψ.M.(newparticles, [Random.GLOBAL_RNG], [p+1], [particle], [nothing])
        logpot += logsumexp(particle.β₁ .* Gψ.ψ[p+1](newparticles, :log)) - log(Gψ.Nₘ)
    end

    if p == 1
        Gψ.M.(newparticles, [Random.GLOBAL_RNG], [1], [particle], [nothing]) # doesn't use particle, overrides newparticles
        logpot += logsumexp(particle.β₀ .* Gψ.ψ[1](newparticles, :log)) - log(Gψ.Nₘ)
    end

    return logpot
end