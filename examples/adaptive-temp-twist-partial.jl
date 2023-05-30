mutable struct ExpTilt{R<:Real} <: AbstractTwist
    h::AbstractVector{R} # exp(x'h) .. same scale as potential vector in MvNormalCanon
end

ExpTilt{R}(d::Int64) where {R<:Real} = ExpTilt(zeros(R,d))

function (λ::ExpTilt{R})(x::AbstractVector{R}, outscale::Symbol) where {R<:Real}
    ℓval = x' * λ.h
    if outscale == :log
        return ℓval
    else
        @warn "Returning on standard scale, logscale = $ℓpdf"
        exp(ℓval)
    end
end

tilt(d::MvNormal, λ::ExpTilt{R}) where {R<:Real} = MvNormal(d.μ + d.Σ*λ.h, d.Σ)

Base.:/(ψ::ExpQuadTwist{R}, λ::ExpTilt{R}) where {R<:Real} = ExpQuadTwist(ψ.h - λ.h, ψ.J)

struct TwistDecomp{R<:Real} <: AbstractTwist # particle stores this
    Mλ::Sampleable{<:VariateForm,<:ValueSupport}
    λ::ExpTilt{R}
    r::ExpQuadTwist{R}
    β::Float64
    logZMλ::Float64
end

function (twist::TwistDecomp{R})(x::AbstractVector{R}, outscale::Symbol) where {R<:Real}
    ℓval = twist.λ(x, :log) + twist.β * twist.r(x, :log)
    if outscale == :log
        return ℓval
    else
        @warn "Returning on standard scale, logscale = $ℓpdf"
        exp(ℓval)
    end

end

function logmcmeantwist(rng, twist::TwistDecomp{R}, Nₘ::Int64) where {R<:Real}

    x = rand(rng, twist.Mλ, Nₘ) # ~ Mₚ^λ(xₚ₋₁, ⋅)
    return twist.logZMλ + logsumexp(twist.β .* twist.r(x, :log)) - log(Nₘ)
    # estimate of Mₚ₊₁(ψₚ₊₁)(xₚ) = Mₚ₊₁(λₚ₊₁)(xₚ)Mₚ₊₁^λ(rₚ₊₁)(xₚ)
end

struct DecompTemperAdaptSampler{T<:AbstractTwist} <: Sampleable{Univariate, Continuous}
    Mλ::Sampleable{<:VariateForm,<:ValueSupport} # partial
    λ::ExpTilt{<:Real}
    r::T
    logZMλ::Float64
    logα::Float64 # acceptance target
    Nₐ::Int64 # sample to estimate acceptance rate
end

# Particle Structure #
mutable struct DecompTwistVectorParticle{d} <: AbstractParticle # TT = Tempered Twist by β
    x::SVector{d, Float64} # current: t
    twₚ₊₁::TwistDecomp{Float64} # next: t + 1
    logψ̃::Float64 # ψ̃ = Mₚ₊₁(ψₚ₊₁)(xₚ) / ψₚ(xₚ)
    DecompTwistVectorParticle{d}() where d = new()
end

# twisting functions markov kernel
struct DecompTemperKernel{T<:AbstractTwist}
    logα::Float64 # acceptance target
    Nₐ::Int64 # sample to estimate acceptance rate
end

function (K::DecompTemperKernel{T})(d::AbstractMvNormal, ψ::T) where {R<:Real, T<:ExpQuadTwist{R}}
    # best partial analytical
    b = (inv(ψ.J) + cov(d)) \ ((ψ.J \ ψ.h) - mean(d))
    λ = ExpTilt(b)
    r = ψ / λ
    Mλ = tilt(d, λ)
    logZMλ = (b' * cov(d) * b)/2 + mean(d)' * b
    return DecompTemperAdaptSampler(Mλ, λ, r, logZMλ, K.logα, K.Nₐ)
end

function Base.rand(rng::AbstractRNG, s::DecompTemperAdaptSampler{ExpQuadTwist{R}}) where {R<:Real}
        
    # choose β from trial draws
    logrx = s.r(rand(rng, s.Mλ, s.Nₐ), :log)

    # define reach_acc_rate(b) > 0 if accept target is exceeded
    reach_acc_rate(b::Float64) = logsumexp(b .* logrx) - log(s.Nₐ) - s.logα
    if reach_acc_rate(1.0) > 0
        β = 1.0
    else
        β = find_zero(reach_acc_rate, (0,1))
    end

    return TwistDecomp(s.Mλ, s.λ, s.r, β, s.logZMλ)
end


# Twisted Distribution

struct DecompTwistedMarkovChain{D<:Sampleable,K<:MarkovKernel,T<:AbstractTwist} <: AbstractMarkovChain
    μ::D
    M::K
    λK::DecompTemperKernel{T}
    n::Int64
    ψ::AbstractVector{T}
    Nₘ::Int64 # MC repeats for twisted potential estimate
end

PathTwister.untwist(chain::DecompTwistedMarkovChain) = MarkovChain(chain.μ, chain.M, chain.n)

function Base.getindex(chain::DecompTwistedMarkovChain{D,K,T}, i::Integer; base::Bool = false) where {D<:Sampleable,K<:MarkovKernel,T<:AbstractTwist}
    
    if !base
        # twisted M, base = false
        if i == 1
            return (twₚ) -> RejectionSampler(twₚ.Mλ, twₚ.r, twₚ.β, MAXITER)
        elseif i <= length(chain)
            return (old) -> RejectionSampler(old.twₚ₊₁.Mλ, old.twₚ₊₁.r, old.twₚ₊₁.β, MAXITER)
        else
            @error "Index $i not defined for chain."
        end
    else 
        # untwisted M, base = true
        if i == 1
            return chain.μ
        elseif i <= length(chain)
            return (chain.M isa AbstractVector) ? chain.M[i-1] : chain.M
        else
            @error "Index $i not defined for chain."
        end
    end
    
end

function (chain::DecompTwistedMarkovChain{D,K,T})(new::DecompTwistVectorParticle, rng, p::Int64, old::DecompTwistVectorParticle, ::Nothing) where {D<:Sampleable,K<:MarkovKernel,T<:AbstractTwist}
    
    new.logψ̃ = 0.0 # MC estimate of ψ̃ = Mₚ₊₁(ψₚ₊₁)(xₚ) / ψₚ(xₚ) for p > 1.  For p == 1 includes M₁(ψ₁) factor also

    # calculate decomp (repeated per particle - inefficient for p = 1)
    if p == 1
        λKₚ = chain.λK(chain[1, base = true], chain.ψ[1])
        twₚ = rand(rng, λKₚ)
        new.logψ̃ += logmcmeantwist(rng, twₚ, chain.Nₘ)
    else # recall decomp
        twₚ = old.twₚ₊₁
    end
   
    # function changes! new::DecompTwistVectorParticle
    Mψx = (p == 1) ? chain[1](twₚ) : chain[p](old) 
    # chain[p] == twisted mutation/distribution at p
    # old doesn't exist at chain[1]
    # need old.twₚ₊₁, use new.twₚ instead
    
    # mutate: x
    xₚ = Array{eltype(Mψx)}(undef, size(Mψx))
    rand!(rng, Mψx, xₚ)
    new.x = xₚ

    # mutate: next twist
    if p < length(chain)
        λKₚ₊₁ = chain.λK(chain[p+1, base = true](xₚ), chain.ψ[p+1])
        new.twₚ₊₁ = rand(rng, λKₚ₊₁)
        new.logψ̃ += logmcmeantwist(rng, new.twₚ₊₁, chain.Nₘ)
    end

    new.logψ̃ -= twₚ(xₚ, :log) # divide by ψₚ

end

# Twisted Potential Structure
struct MCDecompTwistedLogPotentials <: LogPotentials
    G::LogPotentials # original potentials
end

PathTwister.untwist(Gψ::MCDecompTwistedLogPotentials) = Gψ.G

# needs to depend on rng in final version... (only serial for now)
function (Gψ::MCDecompTwistedLogPotentials)(p::Int64, particle::DecompTwistVectorParticle, ::Nothing)

    return logpdf(Gψ.G.obs[p], value(particle)) + particle.logψ̃
    # ψ̃ = Mₚ₊₁(ψₚ₊₁)(xₚ) / ψₚ(xₚ) for p > 1.  For p == 1 includes M₁(ψ₁) factor also

end