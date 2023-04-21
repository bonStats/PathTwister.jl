
abstract type AbstractTwist end

abstract type AbstractParticle end
value(particle::AbstractParticle) = particle.x
value(particles::Vector{P}) where {P<:AbstractParticle} = reduce(hcat, value.(particles))

abstract type MarkovKernel end
(M::MarkovKernel)(x::AbstractParticle) = M(value(x))

abstract type AbstractMarkovChain <:Function end
function (chain::AbstractMarkovChain)(new::AbstractParticle, rng, p::Int64, old::AbstractParticle, ::Nothing)
    # warning changes! new::AbstractParticle
    d = (p == 1) ? chain[1] : chain[p](old)
    new.x = rand(rng, isa(d, Sampleable) ? d : d())
    # rand!(rng, isa(d, Sampleable) ? d : d(), new.x) # use this instead?
end

abstract type LogPotentials <:Function end