# flat twist for testing twisting algorithms
struct FlatTwist{R<:Real} <: AbstractTwist
    logvalue::Float64
end

FlatTwist(R::DataType) = FlatTwist{R}(zero(R))

function (ψ::FlatTwist{R})(x::AbstractVector{R}, outscale::Symbol) where {R<:Real}
    ℓpdf = ψ.logvalue
    if outscale == :log
        return ℓpdf
    else
        @warn "Returning on standard scale, logscale = $ℓpdf"
        exp(ℓpdf)
    end
end

function (ψ::FlatTwist{R})(x::AbstractMatrix{R}, outscale::Symbol) where {R<:Real}
    ℓpdf = repeat([ψ.logvalue], size(x,2))
    if outscale == :log
        return ℓpdf
    else
        @warn "Returning on standard scale, logscale[1] = $(ℓpdf[1])"
        exp.(ℓpdf)
    end
end

function (ψ::FlatTwist{R})(particles::Vector{<:P}, outscale::Symbol) where {R<:Real, P<:AbstractParticle}
    ℓpdf = repeat([ψ.logvalue], length(particles))
    if outscale == :log
        return ℓpdf
    else
        @warn "Returning on standard scale, logscale[1] = $(ℓpdf[1])"
        exp.(ℓpdf)
    end
end