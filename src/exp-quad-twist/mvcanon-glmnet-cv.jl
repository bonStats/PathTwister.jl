# NOTE: constraints argument only accepts lower < 0 and upper > 0
# therefore, better to start with EigenMvNormalCanon as I
# also constraints are overridden

mutable struct EigenMvNormalCanon{R<:Real}
    J::Eigen{R, R, Matrix{R}, Vector{R}}
    h::Vector{R}
end

EigenMvNormalCanon{R}(d::Int64) where {R<:Real} =  EigenMvNormalCanon(eigen(zeros(R,d,d)+I), zeros(R,d))

# Q Λ Q'
evalues(emvn::EigenMvNormalCanon{R}) where {R<:Real} = emvn.J.values
evectors(emvn::EigenMvNormalCanon{R}) where {R<:Real} = emvn.J.vectors

# D(σ) R D(σ)
σvalues(emvn::EigenMvNormalCanon{R}) where {R<:Real} = [sqrt(dot(rx .* evalues(emvn), rx)) for rx in eachrow(evectors(emvn))]
function ρvectors(emvn::EigenMvNormalCanon{R}, σvals::AbstractVector{R}) where {R<:Real}
    scaledQ = Diagonal(σvals) \ evectors(emvn)
    # R = D(σ)⁻¹ (Q Λ Q') D(σ)⁻¹
    return scaledQ * Diagonal(evalues(emvn)) * scaledQ'
end
ρvectors(emvn::EigenMvNormalCanon{R}) where {R<:Real} = ρvectors(emvn, σvalues(emvn))

#J
prec(emvn::EigenMvNormalCanon{R}) where {R<:Real} = evectors(emvn) * Diagonal(evalues(emvn)) * evectors(emvn)'
#h 
linear(emvn::EigenMvNormalCanon{R}) where {R<:Real} = emvn.h

"""
    learn_mvcanon_cvnet(x::AbstractMatrix{R}, y::AbstractVector{R})

x = linear predictors
y = response
"""
function learn_mvcanon_cvnet(x::AbstractMatrix{R}, y::AbstractVector{R}, folds::AbstractVector{Int64}, iter::Int64, diagϵ::Float64=1e-02, corrϵ::Float64=1e-01; alpha::Float64 = 1.0) where {R<:Real}
    
    d = size(x, 2)
    emvn = EigenMvNormalCanon{R}(d)

    # scratch space for regression setup (linear unchanged, quad and cross change)
    X = zeros(m, d*(d-1)÷2 + 2*d)
    lincols = 1:d
    quadcols = (d+1):(2*d)
    crosscols = (2*d+1):size(X,2)

    # set linear
    X[:,lincols] = x

    # initialise diagonal
    learn_mvcanon_cvnet_eigen!(emvn, X, quadcols, lincols, y, folds, diagϵ, alpha)

    for _ in 1:iter
        learn_mvcanon_cvnet_pcor!(emvn, X, crosscols, lincols, y, folds, diagϵ, corrϵ, alpha)
        learn_mvcanon_cvnet_eigen!(emvn, X, quadcols, lincols, y, folds, diagϵ, alpha)
    end

    return emvn
end

# auto dispatch for folds::Int64 and folds::Vector
_glmnetcv(X::AbstractMatrix, y::Vector, folds::Int64, constraints::AbstractMatrix, alpha::Float64) = glmnetcv(X, y, nfolds = folds, constraints = constraints, alpha = alpha)
_glmnetcv(X::AbstractMatrix, y::Vector, folds::AbstractVector{Int64}, constraints::AbstractMatrix, alpha::Float64) = glmnetcv(X, y, folds = folds, constraints = constraints, alpha = alpha)

# Problem:
# y ~ -0.5(x-μ)ᵀP(x-μ) = -0.5xᵀPx + xᵀPμ = -0.5xᵀJx + xᵀh + c
# y ~ -0.5∑aᵢᵢxᵢ² -∑aᵢⱼxᵢxⱼ1(i<j) + ∑bᵢxᵢ
# e.g. quadratic part in two dimensions:
# |x₁|ᵀ * |a b| * |x₁|   | ax₁ + bx₂|ᵀ |x₁|   
# |x₂|    |b c|   |x₂| = | bx₁ + cx₂|  |x₂| = ax₁² + 2bx₁x₂ + cx₂²

# Strategy
# precision matrix has 2 useful decompositions
# (i) P = √D R √D, where Q is partial correlation matrix
# (ii) P = Q Λ Q', eigen decomposition
# In (i) components of R are linear for regression. With simple constraints Rᵢᵢ = 1, Rᵢⱼ ∈ (-1,1) 
# In (ii) components of Λ are linear for regression. With simple constraints Λᵢᵢ > 0, Λᵢⱼ = 0.
# simply iterate between two regressions

function learn_mvcanon_cvnet_eigen!(emvn::EigenMvNormalCanon{R}, X::AbstractMatrix{R}, quadXcols::UnitRange{Int64}, linXcols::UnitRange{Int64}, yf::AbstractVector{R}, folds::Union{AbstractVector{Int64}, Int64}, ϵ::Float64, alpha::Float64) where {R<:Real}
    
    quadX = @view X[:,quadXcols]
    linX = @view X[:,linXcols]
    d = size(linX, 2)
    
    # scaled X (linear only, used as scratch)
    quadX .= linX * evectors(emvn)

    # -xᵢ' J xᵢ / 2 + h'xᵢ (X contains evectors)
    logψx = [-dot(x .* evalues(emvn), x)/2 + dot(emvn.h, linX[i,:]) for (i, x) in enumerate(eachrow(quadX))]
    
    # residual y
    y = yf - logψx

    # full X
    quadX .= -(quadX .^2) ./ 2
    fullX = @view X[:, union(quadXcols,linXcols)]

    constraints = repeat([-Inf; Inf], 1, 2*d)
    constraints[1,1:d] = - evalues(emvn) .+ ϵ

    # run (warning: scales constraints)
    res = _glmnetcv(fullX, y, folds, constraints, alpha)

    emvn.J = Eigen(emvn.J.values + coef(res)[1:d], emvn.J.vectors)
    emvn.h = emvn.h + coef(res)[(d+1):end]

    @assert all(emvn.J.values .> 0.0)

end


function learn_mvcanon_cvnet_pcor!(emvn::EigenMvNormalCanon{R}, X::AbstractMatrix{R}, crossXcols::UnitRange{Int64}, linXcols::UnitRange{Int64}, yf::AbstractVector{R}, folds::Union{AbstractVector{Int64}, Int64}, evϵ::Float64, corrϵ::Float64, alpha::Float64) where {R<:Real}
    
    crossX = @view X[:,crossXcols]
    linX = @view X[:,linXcols]
    d = size(linX, 2)
    
    # -xᵢ' J xᵢ / 2 + h'xᵢ
    J = prec(emvn)
    logψx = [-dot(x, J * x)/2 + dot(emvn.h, x) for x in eachrow(linX)] 
    
    # residual y
    y = yf - logψx

    # scaled X (cross only) 
    σvals = σvalues(emvn)
    crossX .= -hcat([linX[:,cc[1]] .* linX[:,cc[2]] .* (σvals[cc[1]] * σvals[cc[2]]) for (_, cc) in uppertriiter(d)]...)
    
    # full X
    fullX = @view X[:, union(crossXcols,linXcols)]

    ρvecs = ρvectors(emvn, σvals)
    linconstraints = repeat([-Inf; Inf], 1, d)
    crossconstraints = hcat([[-1+corrϵ;1-corrϵ] .- ρvecs[cc[1],cc[2]] for (_, cc) in uppertriiter(d)]...)
    constraints = [crossconstraints linconstraints]

    # run (warning: scales constraints)
    res = _glmnetcv(fullX, y, folds, constraints, alpha)

    J = zeros(d,d)
    J[diagind(J)] = σvals .^2
    for (i, cc) in uppertriiter(d)
        J[cc] = J[reverse(cc)] = (coef(res)[i] + ρvecs[cc]) * σvals[cc[1]] * σvals[cc[2]]
    end
    J = Symmetric(J)

    emvn.J = eigen!(J) # mutates J
    emvn.h = emvn.h + coef(res)[(size(crossX,2)+1):end]

    if any(emvn.J.values .<= 0.0) # correction to keep numerically positive definite.
        emvn.J = Eigen( max.(evalues(emvn), evϵ), evectors(emvn))
        # This projection is the closest...
        # https://nhigham.com/2021/01/26/what-is-the-nearest-positive-semidefinite-matrix/
    end

end

# helper functions:

# uppertri iterator
uppertriiter(d::Int64) = enumerate(Iterators.filter(c -> c[1] < c[2], CartesianIndices((1:d,1:d))))

# reverse coordinates of CartesianIndex
Base.reverse(cc::CartesianIndex) = CartesianIndex(cc[2], cc[1])



# using Distributions
# using GLMNet
# m = 2000
# d = 25
# linX = rand(m,d)

# quad = [1., 2., 0.5]
# yf = linX[:,1] .- (0.5 * quad[1] * linX[:,1] .^2) .- (0.5 * quad[2] * linX[:,2].^2) .- (0.5 * quad[3] * linX[:,3] .^2) .- 0.9*(linX[:,1] .* linX[:,2]) .+ rand(Normal(0.0,0.1), m)
# folds = rand(Categorical(6), m)

# res = learn_mvcanon_cvnet(linX, yf, folds, 1, alpha = 1.0)
# prec(res)
# linear(res)