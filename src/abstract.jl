abstract type AbstractParticle end
abstract type AbstractTwist end

# fallback...
value(particle::AbstractParticle) = particle.x
value(particles::Vector{P}) where {P<:AbstractParticle} = reduce(hcat, value.(particles))
