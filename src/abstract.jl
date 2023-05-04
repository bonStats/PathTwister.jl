
abstract type AbstractTwist end

abstract type AbstractParticle end
value(particle::AbstractParticle) = particle.x
value(particles::Vector{P}) where {P<:AbstractParticle} = reduce(hcat, value.(particles))

abstract type MarkovKernel end
(M::MarkovKernel)(x::AbstractParticle) = M(value(x))
(M::MarkovKernel)(rng, x::AbstractParticle) = M(rng, value(x))

abstract type AbstractMarkovChain <:Function end

# default methods:
function (chain::AbstractMarkovChain)(new::AbstractParticle, rng, p::Int64, old::AbstractParticle, ::Nothing)
    # warning changes! new::AbstractParticle
    d = (p == 1) ? chain[1] : chain[p](old)
    new.x = rand(rng, isa(d, Sampleable) ? d : d())
    # rand!(rng, isa(d, Sampleable) ? d : d(), new.x) # use this instead?
end

function Base.length(chain::AbstractMarkovChain)
    @assert hasfield(typeof(chain), :μ) "Default methods for T<:AbstractMarkovChain require field for initial distribution named μ. Implement Base.length and Base.getindex for T."
    @assert hasfield(typeof(chain), :M) "Default methods for T<:AbstractMarkovChain require field for Markov Kernel(s) M. Implement Base.length and Base.getindex for T."
    
    hasfield(typeof(chain), :n) ? chain.n : 1 + length(chain.M)
    
end

function Base.getindex(chain::AbstractMarkovChain, i::Integer)
   @assert 1 <= i <= length(chain)
   if i == 1
    return chain.μ
   else
    return (chain.M isa AbstractVector) ? chain.M[i-1] : chain.M
   end
end

abstract type LogPotentials <:Function end