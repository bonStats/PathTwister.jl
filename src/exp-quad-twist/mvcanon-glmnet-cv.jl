# NOTE: constraints argument only accepts lower < 0 and upper > 0
# therefore, better to start with EigenMvNormalCanon as I
# also constraints are overriddem

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
linear(emvn::EigenMvNormalCanon{R}) where {R<:Real} = emvn.h

"""
    learn_mvcanon_cvnet(x::AbstractMatrix{R}, y::AbstractVector{R})

x = linear predictors
y = response
"""
function learn_mvcanon_cvnet(x::AbstractMatrix{R}, y::AbstractVector{R}, quad::Union{AbstractVector{R}, Nothing}, cross::Union{AbstractVector{R}, Nothing}, ϵ::Float64=1e-04) where {R<:Real}

    if isnothing(quad)
        quadlower = zero(x[1,:]) .+ ϵ
    else
        quadlower = -quad .+ ϵ
    end

    # n = size(x,1)
    d = size(x,2)
    qX = -0.5 .* (x .^ 2)
    qcon = [[quadlower repeat([Inf], d)]' repeat([-Inf; Inf], 1, d)]
    # assume aᵢⱼ = 0
    qres = glmnetcv([qX x], y, folds = folds, constraints = qcon, alpha = 0.0)
    qcoef = coef(qres)[1:d]

    # scaled -∑aᵢⱼxᵢxⱼ1(i<j) * √ aᵢᵢ aⱼⱼ # CHECK THIS
    cX = [ - x[:,cc[1]] .* x[:,cc[2]] .* sqrt(qcoef[cc[1]] * qcoef[cc[2]]) for (_, cc) in uppertriiter(d)]
    dc = size(cX, 2)

    if isnothing(cross)
        cbounds = repeat([-1; 1], 1, dc)
    else
        cbounds = [cross cross]'
    end

    ccon = [ repeat([-Inf; Inf], 1, dc)]
    res = glmnetcv([cX x], y, folds = folds, constraints = ccon, alpha = 1.0)
end

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

function learn_mvcanon_cvnet_eigen!(emvn::EigenMvNormalCanon{R}, linX::AbstractMatrix{R}, yf::AbstractVector{R}, folds, ϵ::Float64=1e-04, alpha::Float64 = 0.0) where {R<:Real}
    
    d = size(linX, 2)
    
    # scaled X (linear only)
    X = linX * evectors(emvn)

    # -xᵢ' J xᵢ / 2 + h'xᵢ (X contains evectors)
    logψx = [-dot(x .* evalues(emvn), x)/2 + dot(emvn.h, linX[i,:]) for (i, x) in enumerate(eachrow(X))]
    
    # residual y
    y = yf - logψx

    # full X
    X = [-(X .^2) ./ 2 linX]

    constraints = repeat([-Inf; Inf], 1, 2*d)
    constraints[1,1:d] = - evalues(emvn) .+ ϵ

    # run (warning: scales constraints)
    res = glmnetcv(X, y, folds = folds, constraints = constraints, alpha = alpha)

    emvn.J = Eigen(emvn.J.values + coef(res)[1:d], emvn.J.vectors)
    emvn.h = emvn.h + coef(res)[(d+1):end]

    @assert all(emvn.J.values .> 0.0)

end


function learn_mvcanon_cvnet_pcor!(emvn::EigenMvNormalCanon{R}, crossX::AbstractMatrix{R}, linX::AbstractMatrix{R}, yf::AbstractVector{R}, folds, ϵ::Float64=1e-04, alpha::Float64 = 0.0) where {R<:Real}
    
    d = size(linX, 2)
    
    # -xᵢ' J xᵢ / 2 + h'xᵢ
    J = prec(emvn)
    logψx = [-dot(x, J * x)/2 + dot(emvn.h, x) for x in eachrow(linX)] 
    
    # residual y
    y = yf - logψx

    # scaled X (cross only) 
    σvals = σvalues(emvn)
    crossX .= hcat([linX[:,cc[1]] .* linX[:,cc[2]] .* (σvals[cc[1]] * σvals[cc[2]]) for (_, cc) in uppertriiter(d)]...)
    
    # full X
    X = [-crossX linX]

    ρvecs = ρvectors(emvn, σvals)
    linconstraints = repeat([-Inf; Inf], 1, d)
    crossconstraints = hcat([[-1;1] .- ρvecs[cc[1],cc[2]] for (_, cc) in uppertriiter(d)]...)
    constraints = [crossconstraints linconstraints]

    # run (warning: scales constraints)
    res = glmnetcv(X, y, folds = folds, constraints = constraints, alpha = alpha)

    J = zeros(d,d)
    J[diagind(J)] = σvals .^2
    for (i, cc) in uppertriiter(d)
        J[cc] = J[reverse(cc)] = (coef(res)[i] + ρvecs[cc]) * σvals[cc[1]] * σvals[cc[2]]
    end
    
    emvn.J = eigen(J)
    emvn.h = emvn.h + coef(res)[(size(crossX,2)+1):end]

    @assert all(emvn.J.values .> 0.0)

end

# helper functions:

# uppertri iterator
uppertriiter(d::Int64) = enumerate(Iterators.filter(c -> c[1] < c[2], CartesianIndices((1:d,1:d))))

# reverse coordinates of CartesianIndex
Base.reverse(cc::CartesianIndex) = CartesianIndex(cc[2], cc[1])



# using Distributions
# using GLMNet
# m = 1000
# d = 3
# linX = rand(m,d)

# quad = [1., 2., 0.5]

# yf = linX[:,1] .- (0.5 * quad[1] * linX[:,1] .^2) .- (0.5 * quad[2] * linX[:,2].^2) .- (0.5 * quad[3] * linX[:,3] .^2) .- 0.9*(linX[:,1] .* linX[:,2]) .+ rand(Normal(0.0,0.1), m)

# emvn = EigenMvNormalCanon{Float64}(d)

# ϵ = 1e-04
# alpha = 1.0
# folds = rand(Categorical(4), m)

# X = zeros(m, d*(d-1)÷2 + 2*d)

# linX = @view X[:,(end-d+1):end]
# crossX = @view X[:,(d+1):(end-d)]

# XX = @view [crossX linX]

learn_mvcanon_cvnet_eigen!(emvn, linX, yf, folds, ϵ, alpha)
learn_mvcanon_cvnet_pcor!(emvn, crossX, linX, yf, folds, ϵ, alpha)

prec(emvn)
linear(emvn)