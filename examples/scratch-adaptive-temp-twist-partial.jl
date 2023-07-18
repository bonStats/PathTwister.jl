import StatsFuns: logsumexp
import Roots: find_zero

abstract type AbstractTemperKernel{T<:AbstractTwist} end

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


function (λ::ExpTilt{R})(X::AbstractMatrix{R}, outscale::Symbol) where {R<:Real}
    ℓvals = [x' * λ.h for x in eachcol(X)]
    if outscale == :log
        return ℓvals
    else
        @warn "Returning on standard scale, logscale[1] = $ℓpdf[1]"
        exp.(ℓvals)
    end
end


Base.iszero(λ::ExpTilt{R}) where {R<:Real} = iszero(λ.h)
tilt(d::MvNormal, λ::ExpTilt{R}) where {R<:Real} = iszero(λ) ? d : MvNormal(d.μ + d.Σ*λ.h, d.Σ)
untilt(d::MvNormal, λ::ExpTilt{R}) where {R<:Real} = iszero(λ) ? d : MvNormal(d.μ - d.Σ*λ.h, d.Σ)

Base.:/(ψ::ExpQuadTwist{R}, λ::ExpTilt{R}) where {R<:Real} = ExpQuadTwist(ψ.h - λ.h, ψ.eJ)

Base.:*(β::R, λ::ExpTilt{R}) where {R<:Real} = ExpTilt(β * λ.h)


struct TwistDecomp{R<:Real} <: AbstractTwist # particle stores this
    Mλ::Sampleable{<:VariateForm,<:ValueSupport}
    λ::ExpTilt{R}
    r::ExpQuadTwist{R}
    logZMλ::Float64
end

function (twist::TwistDecomp{R})(x::AbstractVector{R}, outscale::Symbol) where {R<:Real}
    ℓval = twist.λ(x, :log) + twist.r(x, :log)
    if outscale == :log
        return ℓval
    else
        @warn "Returning on standard scale, logscale = $ℓpdf"
        exp(ℓval)
    end

end

function logmcmeantwist(rng, twist::TwistDecomp{R}, Nₘ::Int64) where {R<:Real}

    # THIS: has seems to have higher variance (apparent in high dimensions) since we are trying to minimise
    # Mₚ₊₁(λₚ₊₁)(xₚ) which appears in the denominator of Mₚ₊₁^λ(rₚ₊₁)(xₚ)
    # x = rand(rng, twist.Mλ, Nₘ) # ~ Mₚ^λ(xₚ₋₁, ⋅)
    # return twist.logZMλ + logsumexp(twist.β .* twist.r(x, :log)) - log(Nₘ)
    # # estimate of Mₚ₊₁(ψₚ₊₁)(xₚ) = Mₚ₊₁(λₚ₊₁)(xₚ)Mₚ₊₁^λ(rₚ₊₁)(xₚ)

    x = rand(rng, untilt(twist.Mλ, twist.λ), Nₘ) # ~ Mₚ(xₚ₋₁, ⋅)
    return logsumexp(twist.r(x, :log) .+  twist.λ(x, :log)) - log(Nₘ)
    # estimate of Mₚ₊₁(ψₚ₊₁)(xₚ)
end

struct DecompTemperAdaptSampler{T<:AbstractTwist} <: Sampleable{Univariate, Continuous}
    M::Sampleable{<:VariateForm,<:ValueSupport} # partial
    ψ::T
    b::Function
    logα::Float64 # acceptance target
    Nₐ::Int64 # sample to estimate acceptance rate
end

# Particle Structure #
mutable struct DecompTwistVectorParticle{d} <: AbstractParticle # TT = Tempered Twist by β
    x::SVector{d, Float64} # current: t
    twₚ₊₁::TwistDecomp{Float64} # next: t + 1
    logψ̃::Float64 # ψ̃ = Mₚ₊₁(ψₚ₊₁)(xₚ) / ψₚ(xₚ)
    rsn::Int64 # number of rejection sampler
    DecompTwistVectorParticle{d}() where d = new()
end

# twisting functions markov kernel
struct DecompTemperKernel{T<:AbstractTwist} <: AbstractTemperKernel{T}
    logα::Float64 # acceptance target
    Nₐ::Int64 # sample to estimate acceptance rate
end

function (K::DecompTemperKernel{T})(d::AbstractMvNormal, ψ::T) where {R<:Real, T<:ExpQuadTwist{R}}
    # best partial analytical function (conditional on tempering β)
    b(β::Float64) = ((inv(ψ.eJ) ./ β) + cov(d)) \ ((inv(ψ.eJ) * ψ.h) - mean(d))

    return DecompTemperAdaptSampler(d, ψ, b, K.logα, K.Nₐ)
end

function Base.rand(rng::AbstractRNG, s::DecompTemperAdaptSampler{ExpQuadTwist{R}}) where {R<:Real}
        
    # choose β from trial draws (targets acceptance rate on ψ, not partial)
    logψx = s.ψ(rand(rng, s.M, s.Nₐ), :log)

    # define reach_acc_rate(b) > 0 if accept target is exceeded
    reach_acc_rate(b::Float64) = logsumexp(b .* logψx) - log(s.Nₐ) - s.logα
    if reach_acc_rate(1.0) > 0
        β = 1.0
    else
        β = find_zero(reach_acc_rate, (0,1))
    end

    λ = ExpTilt(s.b(β))
    Mλ = tilt(s.M, λ)
    r = (β * s.ψ) / λ
    logZMλ = (λ.h' * cov(s.M) * λ.h)/2 + mean(s.M)' * λ.h # logZMλ not in use

    return TwistDecomp(Mλ, λ, r, logZMλ) 
end


# Twisted Distribution

struct DecompTwistedMarkovChain{D<:Sampleable,K<:MarkovKernel,T<:AbstractTwist} <: AbstractMarkovChain
    μ::D
    M::K
    λK::AbstractTemperKernel{T}
    n::Int64
    ψ::AbstractVector{T}
    Nₘ::Int64 # MC repeats for twisted potential estimate
end

PathTwister.untwist(chain::DecompTwistedMarkovChain) = MarkovChain(chain.μ, chain.M, chain.n)

function Base.getindex(chain::DecompTwistedMarkovChain{D,K,T}, i::Integer; base::Bool = false) where {D<:Sampleable,K<:MarkovKernel,T<:AbstractTwist}
    
    if !base
        # twisted M, base = false
        if i == 1
            return (twₚ) -> RejectionSampler(twₚ.Mλ, twₚ.r, 1.0, MAXITER) # β = 1.0 since Mλ, r are pre-tempered
        elseif i <= length(chain)
            return (old) -> RejectionSampler(old.twₚ₊₁.Mλ, old.twₚ₊₁.r, 1.0, MAXITER)
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

mutable struct DecompTwistedScratch{d, T<:AbstractTwist}
    λK::DecompTemperAdaptSampler{T}
    twₚ::TwistDecomp{<:Real}
    Mψx::RejectionSampler{<:VariateForm,<:ValueSupport,T}
    xₚ::MVector{d,Float64} 
    function DecompTwistedScratch{d, T}() where {d, T<:AbstractTwist} 
        scratch = new{d,T}()
        scratch.xₚ = MVector{d,Float64}(undef)
        return scratch
    end
end

function (chain::DecompTwistedMarkovChain{D,K,T})(new::DecompTwistVectorParticle{d}, rng, p::Int64, old::DecompTwistVectorParticle{d}, scratch::DecompTwistedScratch{d,T}) where {d, D<:Sampleable,K<:MarkovKernel,T<:AbstractTwist}
    
    new.logψ̃ = 0.0 # MC estimate of ψ̃ = Mₚ₊₁(ψₚ₊₁)(xₚ) / ψₚ(xₚ) for p > 1.  For p == 1 includes M₁(ψ₁) factor also

    # calculate decomp (repeated per particle - inefficient for p = 1)
    if p == 1
        scratch.λK = chain.λK(chain[1, base = true], chain.ψ[1]) # λKₚ
        scratch.twₚ = rand(rng, scratch.λK)
        new.logψ̃ += logmcmeantwist(rng, scratch.twₚ, chain.Nₘ)
    else # recall decomp
        scratch.twₚ = old.twₚ₊₁
    end
   
    # function changes! new::DecompTwistVectorParticle
    scratch.Mψx = (p == 1) ? chain[1](scratch.twₚ) : chain[p](old) 
    # chain[p] == twisted mutation/distribution at p
    # old doesn't exist at chain[1]
    # need old.twₚ₊₁, use new.twₚ instead
    
    # mutate: x
    #scratch.xₚ = Array{eltype(Mψx)}(undef, size(Mψx))
    new.rsn = rand!(rng, scratch.Mψx, scratch.xₚ)
    new.x = scratch.xₚ

    # mutate: next twist
    if p < length(chain)
        scratch.λK = chain.λK(chain[p+1, base = true](scratch.xₚ), chain.ψ[p+1]) # λKₚ₊₁
        new.twₚ₊₁ = rand(rng, scratch.λK)
        new.logψ̃ += logmcmeantwist(rng, new.twₚ₊₁, chain.Nₘ)
    end

    new.logψ̃ -= scratch.twₚ(scratch.xₚ, :log) # divide by ψₚ

end

# Twisted Potential Structure
struct MCDecompTwistedLogPotentials <: LogPotentials
    G::LogPotentials # original potentials
end

PathTwister.untwist(Gψ::MCDecompTwistedLogPotentials) = Gψ.G

function (Gψ::MCDecompTwistedLogPotentials)(p::Int64, particle::DecompTwistVectorParticle, ::DecompTwistedScratch{d,T}) where {d, T<:AbstractTwist}

    return logpdf(Gψ.G.obs[p], value(particle)) + particle.logψ̃
    # ψ̃ = Mₚ₊₁(ψₚ₊₁)(xₚ) / ψₚ(xₚ) for p > 1.  For p == 1 includes M₁(ψ₁) factor also

end



# no decomposition

# twisting functions markov kernel
struct TemperKernel{T<:AbstractTwist} <: AbstractTemperKernel{T}
    logα::Float64 # acceptance target
    Nₐ::Int64 # sample to estimate acceptance rate
end

function (K::TemperKernel{T})(d::AbstractMvNormal, ψ::T) where {R<:Real, T<:ExpQuadTwist{R}}
    # no partial analytical
    b(β::Float64) = zeros(length(d))
    return DecompTemperAdaptSampler(d, ψ, b, K.logα, K.Nₐ)
end