# test with smaller n, check still working, up number of particles
# check auto partial code with new expquad

using SequentialMonteCarlo
using Distributions
using LinearAlgebra
using StaticArrays
using Random
using PathTwister
using Roots
import StatsFuns: logsumexp
import StatsBase: countmap

## Exact normalising constant
using Kalman, GaussianDistributions

const MAXITER = 10^5

N = 2^10
Nmc = 2^0

# Types: VectorParticle, LinearGaussMarkovKernel, MvNormalNoise
include("linear-gauss-hmm.jl")

# Types: TTVectorParticle, TemperAdaptSampler, TemperTwist, AdaptiveTwistedMarkovChain, MCTwistedLogPotentials
include("adaptive-temp-twist.jl")

# Types: ExpTilt, TwistDecomp, DecompTemperAdaptSampler, DecompTwistVectorParticle, DecompTemperKernel, 
# DecompTwistedMarkovChain, MCDecompTwistedLogPotentials
#include("adaptive-temp-twist-partial.jl")
include("scratch-adaptive-temp-twist-partial.jl")

# setup problem
n = 200
d = 3
μ = MvNormal(SMatrix{d,d}(1.0I))

A = @SMatrix [0.42^(abs(i-j)+1) for i = 1:d, j = 1:d]

b = @SVector zeros(d)
Σ = SMatrix{d,d}(1.0I)
M = LinearGaussMarkovKernel(A,b,Σ)

#chain = MarkovChain(μ, repeat([M], n-1))
chain = MarkovChain(μ, M, n)

# setup: sim data
noise = MvNormal(d, 1.0)
latentx = repeat([rand(μ)], n)
y = repeat([latentx[1] + rand(noise)], n)
for p in 2:n
    latentx[p] = rand(M(latentx[p-1]))
    y[p] = latentx[p] + rand(noise)
end

## exact normalising constant
D = Matrix(Diagonal(ones((d,))))  

# Define evolution scheme
Kevo = LinearEvolution(A, Gaussian(b, Σ))

# Define observation scheme
Kobs = LinearObservation(Kevo, D, D)

# Filter
_, truelogZ = kalmanfilter(Kobs, 1 => Gaussian(zeros(d), Matrix(μ.Σ)), 1:n .=> y)


## SMC 
potential = MvNormalNoise(y, 1.0*I)

model = SMCModel(chain, potential, n, VectorParticle{d}, Nothing)

smcio = SMCIO{model.particle, model.pScratch}(N, n, 1, true)

smc!(model, smcio)


bestψ = [ExpQuadTwist{Float64}(d) for _ in 1:model.maxn]

lassocvtwist!(bestψ, smcio, model, 8, cvstrategy = 4)


# locally twisted SMC

Mβ = TemperTwist(log(0.05), Nmc)
chainψ = AdaptiveTwistedMarkovChain(μ, M, Mβ, n, bestψ)

potentialψ = MCTwistedLogPotentials(potential, chain, bestψ, Nmc)

modelψ = SMCModel(chainψ, potentialψ, n, TTVectorParticle{d}, Nothing)

smcioψ = SMCIO{modelψ.particle, modelψ.pScratch}(N, n, 1, true)


smc!(model, smcio)
smc!(modelψ, smcioψ)

[smcio.esses smcioψ.esses]

smcio.logZhats[end] - truelogZ
smcioψ.logZhats[end] - truelogZ

countmap(smcio.eves)
countmap(smcioψ.eves)


# check working...

flatψ = repeat([PathTwister.FlatTwist(Float64)], n)
chainψflat = AdaptiveTwistedMarkovChain(μ, M, TemperTwist(log(0.05), 1), n, flatψ)
potentialψflat = MCTwistedLogPotentials(potential, chain, flatψ, 1)

modelψflat = SMCModel(chainψflat, potentialψflat, n, TTVectorParticle{d}, Nothing)

smcioψflat = SMCIO{modelψflat.particle, modelψflat.pScratch}(N, n, 1, true)

smc!(modelψflat, smcioψflat)


smcio.logZhats[end]
smcioψ.logZhats[end]
smcioψflat.logZhats[end]

SequentialMonteCarlo.V(smcio, (x) -> 1, true, false, n)
SequentialMonteCarlo.V(smcioψ, (x) -> 1, true, false, n)

bestψ2 = copy(bestψ)

lassocvtwist!(bestψ2, smcioψ, modelψ, 8, cvstrategy = 4)


chainψ2 = AdaptiveTwistedMarkovChain(μ, M, Mβ, n, bestψ2)
potentialψ2 = MCTwistedLogPotentials(potential, chain, bestψ2, Nmc)

modelψ2 = SMCModel(chainψ2, potentialψ2, n, TTVectorParticle{d}, Nothing)
smcioψ2 = SMCIO{modelψ.particle, modelψ.pScratch}(N, n, 1, true)

smc!(modelψ2, smcioψ2)

smcioψ2.logZhats[end] - truelogZ

bestψ3 = deepcopy(bestψ2)

lassocvtwist!(bestψ3, smcioψ2, modelψ2, 4, cvstrategy = 4)

chainψ3 = AdaptiveTwistedMarkovChain(μ, M, Mβ, n, bestψ3)
potentialψ3 = MCTwistedLogPotentials(potential, chain, bestψ2, Nmc)

modelψ3 = SMCModel(chainψ3, potentialψ3, n, TTVectorParticle{d}, Nothing)
smcioψ3 = SMCIO{modelψ.particle, modelψ.pScratch}(N, n, 1, true)

smc!(modelψ3, smcioψ3)

smcioψ3.logZhats[end] - truelogZ


map(s -> minimum(s.esses), [smcio, smcioψ, smcioψ2, smcioψ3])

map(s -> length(countmap(s.eves)), [smcio, smcioψ, smcioψ2, smcioψ3])

map(s -> s.logZhats[end], [smcio, smcioψ, smcioψ2, smcioψ3]) .- truelogZ

map(s -> SequentialMonteCarlo.V(s, x -> 1, true, false, n), 
[smcio, smcioψ, smcioψ2, smcioψ3])


map(s -> minimum([mean(getfield.(s.allZetas[i], :β₁)) for i in 1:n]), [smcioψ, smcioψ2, smcioψ3])


# partial twist not working in higher dimensions e.g. d = 10
# the code ot find λ may be unstable, or the J matrices ill-conditioned
# despite ensuring that they are Positive Def

# Other options include:
# - Regularise λ learning process
# - Allowing for just Pos Semi Def, defined on subspace of domain
# - Change ψ to alt form with max 1 (solve λ alternatively... approx ψ with quad first)
# - Just use diagonal version of J for quad ψ

Nmc= 8

DMβ = DecompTemperKernel{eltype(bestψ)}(log(0.05), Nmc)
Dchainψ = DecompTwistedMarkovChain(μ, M, DMβ, n, bestψ3, Nmc)

Dpotentialψ = MCDecompTwistedLogPotentials(potential)

#Dmodelψ = SMCModel(Dchainψ, Dpotentialψ, n, DecompTwistVectorParticle{d}, Nothing)
Dmodelψ = SMCModel(Dchainψ, Dpotentialψ, n, DecompTwistVectorParticle{d}, DecompTwistedScratch{d, eltype(bestψ)})

Dsmcioψ = SMCIO{Dmodelψ.particle, Dmodelψ.pScratch}(N, n, 1, true)

@time smc!(Dmodelψ, Dsmcioψ)

Dsmcioψ.logZhats[end] .- truelogZ


D0Mβ = TemperKernel{eltype(bestψ)}(log(0.05), Nmc)
D0chainψ = DecompTwistedMarkovChain(μ, M, D0Mβ, n, bestψ2, Nmc)

D0modelψ = SMCModel(D0chainψ, Dpotentialψ, n, DecompTwistVectorParticle{d}, DecompTwistedScratch{d, eltype(bestψ)})

D0smcioψ = SMCIO{D0modelψ.particle, D0modelψ.pScratch}(N*10, n, 1, true)

smc!(D0modelψ, D0smcioψ)

D0smcioψ.logZhats[end] .- truelogZ


map(s -> minimum(s.esses), [smcio, smcioψ, smcioψ2, smcioψ3, Dsmcioψ])
map(s -> s.logZhats[end], [smcio, smcioψ, smcioψ2, smcioψ3, Dsmcioψ]) .- truelogZ
map(s -> SequentialMonteCarlo.V(s, x -> 1, true, false, n), 
[smcio, smcioψ, smcioψ2, smcioψ3, Dsmcioψ])

minimum([mean(getfield.(getfield.(Dsmcioψ.allZetas[i], :twₚ₊₁),:β)) for i in 1:n])


i = 20; map(s -> (s[i].J \ s[i].h, s[i].J), [bestψ, bestψ2, bestψ3])


smc!(model, smcio); smcio.logZhats[end] - truelogZ

# test scratch: change matrics to static
# change components to static matrices/vectors

# monitor rejection sampler (alter return type)

# recreate PhD experiments

# learned ψ decision when to set to zero? 
# special ψ class, λ for this


A = bestψ[10].J

function ensure_psd_eigen!(X::AbstractMatrix{R}, s::Float64) where {R<:Real}
    ei = eigen(X)
    posvals = ei.values[ei.values .> 0.0]
    minposval = isempty(posvals) ? s : minimum(posvals)
    newvals = map( x -> (x < minposval) ? s*minposval : x, ei.values)
    X .= ei.vectors * Diagonal(newvals) * ei.vectors'
    @warn "PSD ψ correction. Number of negative eigenvalues = $(size(X,2) - length(posvals))"
end

ensure_psd_eigen!(A, 0.1)

X = A


X = LinearAlgebra.symmetric(rand(3,3), :L)

ei = eigen(X)
ei.values[1] = 0.0

(ei.vectors * Diagonal(ei.values) * ei.vectors' - X ) ./ X



A = rand(3,3)
A = A' * A

D = sqrt.(Diagonal(A))

D \ A / D

A / Diagonal(A)

# precision matrix has decomposition A = √D Q √D Where Q is partial correlation matrix
# (1) estimate D by considering x'Dx + b x, D diagonal
# (2) estimate Q,b by considering (x'√D⁻¹) Q (√D⁻¹x) + b x with [-1,1] constraints on Q

# then look into stability of λ