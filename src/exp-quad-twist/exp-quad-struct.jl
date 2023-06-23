
mutable struct ExpQuadTwist{R<:Real} <: AbstractTwist
    h::AbstractVector{R}
    J::AbstractMatrix{R}
    eJ::Eigen{R, R, Matrix{R}, Vector{R}}
    ℓmax::Float64 # maximum value (log scale)
end

unnormalisedmax(h::AbstractVector{R}, eJ::Eigen{R, R, Matrix{R}, Vector{R}}) where {R<:Real} = 0.5 * h' * inv(eJ) * h

# initialise with eigen
function ExpQuadTwist(h::AbstractVector{R}, eJ::Eigen{R, R, Matrix{R}, Vector{R}}) where {R<:Real}
    J = Symmetric(Matrix(eJ))
    ExpQuadTwist(h, J, eJ, unnormalisedmax(h, eJ))
end

# initialise with precision
function ExpQuadTwist(h::AbstractVector{R}, J::AbstractMatrix{R}) where {R<:Real}
    eJ = eigen(J)
    ExpQuadTwist(h, J, eJ, unnormalisedmax(h, eJ))
end

ExpQuadTwist{R}(d::Int64) where {R<:Real} = ExpQuadTwist(zeros(R,d), zeros(R,d,d), eigen(zeros(R,d,d)), zero(R))

# remove?
function evolve!(ψ::ExpQuadTwist{R}, h::AbstractVector{R}, J::AbstractMatrix{R}) where {R<:Real}
    ψ.h = ψ.h + h
    ψ.J = Symmetric(ψ.J + J)
    ψ.eJ = eigen(ψ.J)
    ψ.ℓmax = unnormalisedmax(ψ.h, ψ.eJ)
end

Base.iszero(ψ::ExpQuadTwist{R}) where {R<:Real} = det(ψ.eJ) <= 0.0

eqt(ψ::ExpQuadTwist{R}, x::AbstractVector{R}) where {R<:Real} = dot(x, ψ.h) - 0.5 * dot(x,  ψ.J * x) - ψ.ℓmax # maximum: log(1) = 0
eqt(ψ::ExpQuadTwist{R}, X::AbstractMatrix{R}) where {R<:Real} = [eqt(ψ, x) for x in eachcol(X)] # eachcol consistent with Distributions.jl random number generation

function retlogscale(x::R, logscale::Bool) where {R<:Real}
    if logscale
        return x
    end

    @warn "Returning on standard scale, logscale = $ℓval"
    return exp(x)
end

function retlogscale(x::AbstractVector{R}, logscale::Bool) where {R<:Real}
    if logscale
        return x
    end

    @warn "Returning on standard scale, logscale[1] = $(ℓval[1])"
    return exp.(x)
end

function (ψ::ExpQuadTwist{R})(x::Union{AbstractVector{R},AbstractMatrix{R}}, outscale::Symbol) where {R<:Real}
    ℓval = eqt(ψ, x)
    return retlogscale(ℓval, outscale == :log)
end

(ψ::ExpQuadTwist{R})(particles::Vector{<:P}, outscale::Symbol) where {R<:Real, P<:AbstractParticle} = ψ(value(particles), outscale)


# convert to EigenMvNormalCanon{R<:Real} and back

EigenMvNormalCanon(ψ::ExpQuadTwist{R}) where {R<:Real} = EigenMvNormalCanon(ψ.eJ,ψ.h)
ExpQuadTwist(emvn::EigenMvNormalCanon{R}) where {R<:Real} = ExpQuadTwist(emvn.h, emvn.J)