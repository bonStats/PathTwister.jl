import Distributions: VariateForm, ValueSupport, Univariate, Multivariate, Continuous, rand, rand!, _rand!

function samplebyrejection!(rng::AbstractRNG, s::Sampleable, ψ::AbstractTwist, β::Float64,  maxiter::Int64, x::Union{AbstractVector{R}, DenseMatrix{R}}) where {R<:Real}
    # Assumptions: 
    # 0 ≤ ψ(x) ≤ 1 
    # log scale:returns log ψ(x) 
    # 0 ≤ β ≤ 1

    for i in 1:maxiter
        # x = rand(rng, s)
        rand!(rng, s, x) # doesn't work for static vector?
        if β * ψ(x, :log) > log(rand(rng))
            return i # success
        end
    end

    return 0

end

struct RejectionSampler{V<:VariateForm, S<:ValueSupport, T<:AbstractTwist} <: Sampleable{V,S}
    d::Sampleable{V,S}
    ψ::T
    β::Float64 # inverse temperature
    maxiter::Int64
end

RejectionSampler(d::Sampleable{V,S}, ψ::T; maxiter::Int64) where {V<:VariateForm, S<:ValueSupport, T<:AbstractTwist} = RejectionSampler(d, ψ, 1.0, maxiter)


# rand univariate case:
function rand(rng::AbstractRNG, s::RejectionSampler{Univariate, S, T}) where {S<:ValueSupport, T<:AbstractTwist}
    x = (S <: Continuous) ? 0.0 : 0
    iter = samplebyrejection!(rng, s.d, s.ψ, s.β,  s.maxiter, x)
    @assert iter > 0 "Rejection sample failed to accept after $(s.maxiter) iterations."
    return x
end

# rand/length multivariate-vector case:
Base.length(s::RejectionSampler{Multivariate, S, T}) where {S<:ValueSupport, T<:AbstractTwist} = length(s.d)

function _rand!(rng::AbstractRNG, s::RejectionSampler{Multivariate, S, T}, x::AbstractVector{R}) where {S<:ValueSupport, T<:AbstractTwist, R<:Real}
    iter = samplebyrejection!(rng, s.d, s.ψ, s.β, s.maxiter, x)
    @assert iter > 0 "Rejection sample failed to accept after $(s.maxiter) iterations."
    return x # to work with default Distributions.rand
end

# rand/size multivariate-matrix case:
Base.size(s::RejectionSampler{Multivariate, S, T}) where {S<:ValueSupport, T<:AbstractTwist} = size(s.d)

function _rand!(rng::AbstractRNG, s::RejectionSampler{Multivariate, S, T}, x::DenseMatrix{R}) where {S<:ValueSupport, T<:AbstractTwist, R<:Real}
    iter = samplebyrejection!(rng, s.d, s.ψ, s.β, s.maxiter, x)
    @assert iter > 0 "Rejection sample failed to accept after $(s.maxiter) iterations."
    return x # to work with default Distributions.rand
end

# override default for counting iterations of rejection
function rand!(rng::AbstractRNG, s::RejectionSampler{Multivariate, S, T}, x::AbstractVector{R}) where {S<:ValueSupport, T<:AbstractTwist, R<:Real}
    iter = samplebyrejection!(rng, s.d, s.ψ, s.β, s.maxiter, x)
    @assert iter > 0 "Rejection sample failed to accept after $(s.maxiter) iterations."
    return iter
end