# input SequentialMonteCarlo
# output twisting functions

# note untwist == identity by default

function lassocvtwist!(ψs::Vector{ExpQuadTwist{R}}, smcio::SMCIO{P, S}, model::SMCModel, MCreps::Int64; cvstrategy::Union{Symbol, Int64} = :byreps, quadϵ::Float64 = 1e-02, psdscale::Float64 = 1e-02) where {R<:Real, P<:AbstractParticle, S}
    # TD: how to check if smc!(smcio, model) has been run?
    # TD: Investigate if need scratch for extra particles
    if !smcio.fullOutput
        @error "smcio must have full particle trajectory"
    end

    # setup cv strategy
    if (MCreps > 1) & (cvstrategy == :byreps)
        # folds = id vector which fold an observation belongs to
        folds = vcat([i*ones(Int64, smcio.N) for i in 1:MCreps]...)
    elseif (MCreps > 0) & isa(cvstrategy, Integer)
        # nfolds = how many folds to use
        folds = cvstrategy
    else
        @error "Incompatible MCreps and CV strategy"
    end

    for p in model.maxn:-1:1
        # single design matrix
        X_single, parids = buildpredictors(smcio.allZetas[p])

        if p == model.maxn
            # extra particles for same number of observations over p
            if MCreps > 1
                padparticles = [[model.particle() for _ in 1:smcio.N] for _ in 2:MCreps]
                pushfirst!(padparticles, smcio.allZetas[model.maxn])
                for t in 2:MCreps
                    untwist(model.M!).(padparticles[t], [GLOBAL_RNG], [model.maxn], smcio.allZetas[model.maxn-1], [nothing]) #rng
                end
            else 
                padparticles = [smcio.allZetas[model.maxn]] # always a vector (of Vector{Particle})
            end

            y = vcat([buildresponse(padparticles[i], model, model.maxn) for i in 1:MCreps]...)
            X = vcat(X_single, [buildpredictors(padparticles[i], nothing) for i in 2:MCreps]...)
            
            # adjust regression for current ψ
            if !iszero(ψs[model.maxn])
                y .-= vcat([ψs[model.maxn](padparticles[i], :log) for i in 1:MCreps]...)
            end

        else # p < model.maxn
            y = vcat([buildresponse(smcio.allZetas[p], model, p, ψs[p+1]) for i in 1:MCreps]...) # automatically adjusts for old+new ψ by updating ψs below
            X = repeat(X_single, MCreps) # space efficient version?

            # adjust regression for current ψ
            if !iszero(ψs[p])
                y .-= repeat(ψs[p](smcio.allZetas[p], :log), MCreps)
            end
        end

        h, J = pathcv_hJ(learntwistlassocv(X, y, parids, folds, maxlassoadjust(ψs[p], abs(quadϵ))), parids)

        evolve!(ψs[p], h, J)

        if !isposdef(ψs[p].J)
            ensure_psd_eigen!(ψs[p].J, abs(psdscale))
            refit_linear!(ψs[p], X, y, parids, folds)
        end

    end
end

# build respose variables y = log{ Gₜ(xₜ) Mₜ₊₁(ψₜ₊₁)(xₜ) }
function buildresponse(particles::Vector{<:P}, model::SMCModel, p::Int64, ψ::Union{T, Nothing} = nothing, rng = GLOBAL_RNG) where {P<:AbstractParticle,T<:AbstractTwist}
    ℓresp = untwist(model.lG).([p], particles, nothing) #ℓGp
    if !isnothing(ψ) # when p == n
        newparticles = [model.particle() for i in 1:length(particles)]
        untwist(model.M!).(newparticles, [rng], [p+1], particles, [nothing])
        ℓresp += ψ(newparticles, :log)
    end
    return ℓresp
end

# uppertri iterator
uppertriiter(d::Int64) = enumerate(Iterators.filter(c -> c[1] < c[2], CartesianIndices((1:d,1:d))))

# reverse coordinates of CartesianIndex
Base.reverse(cc::CartesianIndex) = CartesianIndex(cc[2], cc[1])


# build predictor variables X
function buildpredictors(particles::Vector{<:P}) where {P<:AbstractParticle}
    d = length(value(particles[1])) # original predictors
    r = 2*d+binomial(d,2) # regression size (ignore intercept)
    X = ones(length(particles),r)
    # populate
    quadid = 1:d
    linearid = quadid .+ d
    crossid = (max(maximum(linearid),maximum(quadid))+1):r
    X[:,linearid] = value(particles)'
    X[:,quadid] = X[:,linearid] .^2
    for (i, cc) in uppertriiter(d)
        X[:,crossid[i]] = X[:,linearid[cc[1]]] .* X[:,linearid[cc[2]]]
    end
    X, Dict(:linear => linearid, :quad => quadid, :cross => crossid)
end

# build predictor variables X, but don't return ids
function buildpredictors(particles::Vector{<:P}, ::Nothing) where {P<:AbstractParticle}
    X, _ = buildpredictors(particles)
    return X
end

# use CV Lasso to learn predictors with constraint on quadratics
# to make the "precision" PSD
function learntwistlassocv(X::AbstractMatrix{R}, y::AbstractVector{R}, parids::Dict{Symbol, UnitRange{Int64}}, folds::Union{AbstractVector{Int64}, Int64}, quadupper::AbstractVector{R}) where {R<:Real}
    quadconstraint = repeat([-Inf; Inf], 1, size(X, 2))
    quadconstraint[:,parids[:quad]] .= [repeat([-Inf], length(parids[:quad])) quadupper]'
    if isa(folds, AbstractVector)
        return glmnetcv(X, y, folds = folds, constraints = quadconstraint)
    else isa(folds, Integer)
        return glmnetcv(X, y, nfolds = folds, constraints = quadconstraint)
    end
end

# Lasso Regression -> MvNormalCanon form
# ignoring constants...
# ∑aᵢᵢxᵢ² + ∑aᵢⱼxᵢxⱼ1(i<j) + ∑bᵢxᵢ = -0.5(x-μ)ᵀP(x-μ) = -0.5xᵀPx + xᵀPμ = -0.5xᵀJx + xᵀh
# e.g. in two dimensions:
# |x₁|ᵀ * |a b| * |x₁|   | ax₁ + bx₂|ᵀ |x₁|   
# |x₂|    |b c|   |x₂| = | bx₁ + cx₂|  |x₂| = ax₁² + 2bx₁x₂ + cx₂²
function pathcv_hJ(pathcv::GLMNetCrossValidation, parids::Dict{Symbol, UnitRange{Int64}})
    β = coef(pathcv)
    
    h = β[parids[:linear]]
    
    d = length(parids[:linear])
    J = zeros(d,d)
    J[diagind(J)] = -2 .* β[parids[:quad]]
    for (i, cc) in uppertriiter(d)
        J[cc] = J[reverse(cc)] = -β[parids[:cross][i]]
    end

    #h, J: MvNormalCanon
    h, J
end

# maximum adjustment for ψ in terms of constraints
# on quadratic diagonal terms from lassocv
maxlassoadjust(ψ::ExpQuadTwist{R}, ϵ::Float64) where {R<:Real} = 0.5*diag(ψ.J) .- ϵ


# Symmetric -> Symmetric positive definite

function ensure_psd_eigen!(J::AbstractMatrix{R}, sc::Float64) where {R<:Real}
    ei = eigen(J)
    newvals = map( x -> (x < sc) ? sc : x, ei.values)
    # https://nhigham.com/2021/01/26/what-is-the-nearest-positive-semidefinite-matrix/
    J .= symmetric(ei.vectors * Diagonal(newvals) * ei.vectors', :L) # to override elements of X
    @warn "PSD ψ correction. Number of shifted eigenvalues = $(size(J,2) - sum(ei.values .< sc))"
end

function refit_linear!(ψ::ExpQuadTwist{R}, X::AbstractMatrix{R}, y::AbstractVector{R}, parids::Dict{Symbol, UnitRange{Int64}}, folds::Union{AbstractVector{Int64}, Int64}) where {R<:Real}
    ψ.h .= 0.0
    Xlin = X[:, parids[:linear]] # @views?
    res = y .- ψ(Xlin', :log)
    if isa(folds, AbstractVector)
        fitlin = glmnetcv(Xlin, res, folds = folds)
    else isa(folds, Integer)
        fitlin = glmnetcv(Xlin, res, nfolds = folds)
    end
    ψ.h .= coef(fitlin)
end
