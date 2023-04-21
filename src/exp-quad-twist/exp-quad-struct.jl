import Distributions: MvNormalCanon, mode, logpdf

mutable struct ExpQuadTwist{R<:Real} <: AbstractTwist
    h::AbstractVector{R}
    J::AbstractMatrix{R}
end

ExpQuadTwist{R}(d::Int64) where {R<:Real} = ExpQuadTwist(zeros(R,d),zeros(R,d,d))

function evolve!(ψ::ExpQuadTwist{R}, h::AbstractVector{R}, J::AbstractMatrix{R}) where {R<:Real}
    ψ.h = ψ.h + h
    ψ.J = ψ.J + J
end

Base.iszero(ψ::ExpQuadTwist{R}) where {R<:Real} = det(ψ.J) <= 0.0

function (ψ::ExpQuadTwist{R})(x::AbstractVector{R}, outscale::Symbol) where {R<:Real}
    d = MvNormalCanon(ψ.h, ψ.J)
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
    ℓpdf = logpdf.([d], value(particles)) .- logpdf(d, mode(d)) # maximum: log(1) = 0
    if outscale == :log
        return ℓpdf
    else
        @warn "Returning on standard scale, e.g. logscale[1] = $ℓpdf"
        exp.(ℓpdf)
    end
end # NOT SURE WHAT THIS IS FOR, check what happens when we evaluate ψ in rejection sampler etc