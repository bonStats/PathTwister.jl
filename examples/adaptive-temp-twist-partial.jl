mutable struct ExpTilt{R<:Real} <: AbstractTwist
    h::AbstractVector{R} # exp(x'h) .. same scale as potential vector in MvNormalCanon
end

ExpTilt{R}(d::Int64) where {R<:Real} = ExpTilt(zeros(R,d))

tilt(d::MvNormal, λ::ExpTilt{R}) where {R<:Real} = MvNormal(d.μ + d.Σ*λ.h, d.Σ)

Base.:/(ψ::ExpQuadTwist{R}, λ::ExpTilt{R}) where {R<:Real} = ExpQuadTwist(ψ.h - λ.h, ψ.J)

struct TwistDecomp{R<:Real} <: AbstractTwist # particle stores this
    Mλ::Sampleable{<:VariateForm,<:ValueSupport}
    λ::ExpTilt{R}
    r::ExpQuadTwist{R}
    β::Float64
end

struct DecompTemperAdaptSampler{T<:AbstractTwist} <: Sampleable{Univariate, Continuous}
    Mλ::Sampleable{<:VariateForm,<:ValueSupport} # partial
    λ::ExpTilt{<:Real}
    r::ExpQuadTwist{<:Real}
    logZMλ::Float64
    logα::Float64 # acceptance target
    Nₐ::Int64 # sample to estimate acceptance rate
end

# Particle Structure #
mutable struct DecompTwistVectorParticle{d} <: AbstractParticle # TT = Tempered Twist by β
    x::SVector{d, Float64} # current: t
    twₚ::TwistDecomp{Float64} # current: t
    twₚ₊₁::TwistDecomp{Float64} # next: t + 1
    DecompTwistVectorParticle{d}() where d = new()
end

# twisting functions markov kernel
struct DecompTemperKernel{T<:AbstractTwist}
    logα::Float64 # acceptance target
    Nₐ::Int64 # sample to estimate acceptance rate
end

function (K::DecompTemperKernel{ExpQuadTwist{R}})(d::AbstractMvNormal, ψ::ExpQuadTwist{R}) where {R<:Real}
    # best partial analytical
    b = (inv(ψ.J) + cov(d)) \ ((ψ.J \ ψ.h) - mean(d))
    λ = ExpTilt(b)
    r = λ / ψ
    Mλ = tilt(d, λ)
    logZMλ = (b' * cov(d) * b)/2 + mean(d)' * b
    return DecompTemperAdaptSampler(Mλ, λ, r, logZMλ, K.logα, K.Nₐ)
end

function Base.rand(rng::AbstractRNG, s::DecompTemperAdaptSampler{ExpQuadTwist{R}}) where {R<:Real}
        
    # choose β from trial draws
    logrx = s.ψ(rand(rng, s.Mλ, s.Nₐ), :log)

    # define reach_acc_rate(b) > 0 if accept target is exceeded
    reach_acc_rate(b::Float64) = logsumexp(b .* logrx) - log(s.Nₐ) - s.logα
    if reach_acc_rate(1.0) > 0
        β = 1.0
    else
        β = find_zero(reach_acc_rate, (0,1))
    end

    return TwistDecomp(s.Mλ, s.λ, s.r, β)
end

# Twisted Distribution

struct DecompTwistedMarkovChain{D<:Sampleable,K<:MarkovKernel,T<:AbstractTwist} <: AbstractMarkovChain
    μ::D
    M::K
    λK::DecompTemperKernel{T}
    n::Int64
    ψ::AbstractVector{T}
end

PathTwister.untwist(chain::DecompTwistedMarkovChain) = MarkovChain(chain.μ, chain.M, chain.n)

function Base.getindex(chain::DecompTwistedMarkovChain{D,K,T}, i::Integer) where {D<:Sampleable,K<:MarkovKernel,T<:AbstractTwist}
    #twisted M
    if i == 1
        #return (new) -> RejectionSampler(tilt(chain.μ, new.twₚ.λ), new.twₚ.r, new.twₚ.β, MAXITER)
        return (new) -> RejectionSampler(new.twₚ.Mλ, new.twₚ.r, new.twₚ.β, MAXITER)
    elseif i <= length(chain)
        #M = (chain.M isa AbstractVector) ? chain.M[i-1] : chain.M
        #return (old) -> RejectionSampler(tilt(M(old), old.twₚ₊₁.λ), old.twₚ₊₁.r, old.twₚ₊₁.β, MAXITER)
        return (old) -> RejectionSampler(old.twₚ₊₁.Mλ, old.twₚ₊₁.r, old.twₚ₊₁.β, MAXITER)
    else
        @error "Index $i not defined for chain."
    end

end

function Base.getindex(chain::DecompTwistedMarkovChain{D,K,T}, i::Integer, base::Bool = false) where {D<:Sampleable,K<:MarkovKernel,T<:AbstractTwist}

    if !base
        return getindex(chain, i)
    end

    # base = true
    if i == 1
        return chain.μ
    elseif i <= length(chain)
        return (chain.M isa AbstractVector) ? chain.M[i-1] : chain.M
    else
        @error "Index $i not defined for chain."
    end

 end

# function Base.getindex(chain::AdaptiveTwistedMarkovChain{D,K,T}, i::Integer; base::Bool = false) where {D<:Sampleable,K<:MarkovKernel,T<:AbstractTwist}

#     if i == 1

#         if base
#             return chain.μ
#         else 
#             return (new) -> RejectionSampler(tilt(chain.μ, new.twₚ.λ), new.twₚ.r, new.twₚ.β, MAXITER)
#         end

#     elseif i <= length(chain)

#         M = (chain.M isa AbstractVector) ? chain.M[i-1] : chain.M

#         if base
#             return M
#         else
#             return (old) -> RejectionSampler(tilt(M(old), old.twₚ₊₁.λ), old.twₚ₊₁.r, old.twₚ₊₁.β, MAXITER)
#         end

#     else
#         @error "Index $i not defined for chain."
#     end

#  end

function (chain::DecompTwistedMarkovChain{D,K,T})(new::DecompTwistVectorParticle, rng, p::Int64, old::DecompTwistVectorParticle, ::Nothing) where {D<:Sampleable,K<:MarkovKernel,T<:AbstractTwist}
    # calculate decomp (repeated per particle - inefficient)
    if p == 1
        λKₚ = chain.λK(chain[1, base = true], chain.ψ[1])
        new.twₚ = rand(rng, λKₚ)
    end
   
    # function changes! new::DecompTwistVectorParticle
    Mψx = (p == 1) ? chain[1](new) : chain[p](old) 
    # chain[p] == twisted mutation/distribution at p
    # old doesn't exist at chain[1]
    # need old.twₚ₊₁, use new.twₚ instead
    
    # mutate: x
    newx = Array{eltype(Mψx)}(undef, size(Mψx))
    rand!(rng, Mψx, newx)
    new.x = newx

    # mutate: the current twist, just used
    if p > 1
        new.twₚ = old.twₚ₊₁
    end

    # mutate: next twist
    if p < length(chain)
        λKₚ = chain.λK(chain[p+1, base = true](newx), chain.ψ[p+1])
        new.twₚ₊₁ = rand(rng, λKₚ)
    end

end

# Twisted Potential Structure
struct MCDecompTwistedLogPotentials <: LogPotentials
    G::LogPotentials # original potentials
    Nₘ::Int64
end

PathTwister.untwist(Gψ::MCDecompTwistedLogPotentials) = Gψ.G

# needs to depend on rng in final version... (only serial for now)
function (Gψ::MCDecompTwistedLogPotentials)(p::Int64, particle::DecompTwistVectorParticle, ::Nothing)

    logψₚ(x) = particle.twₚ.λ(x, :log) + particle.twₚ.β * particle.twₚ.r(x, :log)

    logpot = logpdf(Gψ.G.obs[p], value(particle)) - logψₚ(value(particle))
    
    if p == length(Gψ.M)
        return logpot
    end
    
    if p < length(Gψ.M)
        Mλ = particle.twₚ₊₁.Mλ #Mₚ₊₁^λ(xₚ, ⋅)
        logZMλ = particle.twₚ₊₁.λlogZMλ
        r = particle.twₚ₊₁.r
        β = particle.twₚ₊₁.β  

        x = rand(Random.GLOBAL_RNG, Mλ, Gψ.Nₘ) # ~ Mₚ₊₁^λ(xₚ, ⋅) # move to rand(DecompTemperKernel,...), combine with logZMλ
        logpot += logsumexp(β .* r.(x, :log)) - log(Gψ.Nₘ)
        logpot += logZMλ
    end

    if p == 1
        Mλ = particle.twₚ.Mλ #Mₚ^λ(xₚ₋₁, ⋅)
        logZMλ = particle.twₚ.λlogZMλ
        r = particle.twₚ.r
        β = particle.twₚ.β  

        x = rand(Random.GLOBAL_RNG, Mλ, Gψ.Nₘ) # ~ Mₚ^λ(xₚ₋₁, ⋅)
        logpot += logsumexp(β .* r.(x, :log)) - log(Gψ.Nₘ)
        logpot += logZMλ
    end

    return logpot
end