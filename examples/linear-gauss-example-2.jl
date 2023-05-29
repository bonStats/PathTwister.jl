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
include("adaptive-temp-twist-partial.jl")

# setup problem
n = 20
d = 2
μ = MvNormal(d, 1.)

A = zeros(d,d)
A[diagind(A)] .= 0.5
A[diagind(A,-1)] .= 0.1
A[diagind(A,1)] .= 0.1

b = zeros(d)
Σ = Matrix(1.0*I, d, d)
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

lassocvtwist!(bestψ, smcio, model, 4, cvstrategy = 8)


# locally twisted SMC

Mβ = TemperTwist(log(0.15), Nmc)
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

bestψ2 = deepcopy(bestψ)

lassocvtwist!(bestψ2, smcioψ, modelψ, 4, cvstrategy = 8)


chainψ2 = AdaptiveTwistedMarkovChain(μ, M, Mβ, n, bestψ2)
potentialψ2 = MCTwistedLogPotentials(potential, chain, bestψ2, Nmc)

modelψ2 = SMCModel(chainψ2, potentialψ2, n, TTVectorParticle{d}, Nothing)
smcioψ2 = SMCIO{modelψ.particle, modelψ.pScratch}(N ÷ 8, n, 1, true)

smc!(modelψ2, smcioψ2)

smcioψ2.logZhats[end] - truelogZ

bestψ3 = deepcopy(bestψ2)

lassocvtwist!(bestψ3, smcioψ2, modelψ2, 4, cvstrategy = 8)

chainψ3 = AdaptiveTwistedMarkovChain(μ, M, Mβ, n, bestψ3)
potentialψ3 = MCTwistedLogPotentials(potential, chain, bestψ2, Nmc)

modelψ3 = SMCModel(chainψ3, potentialψ3, n, TTVectorParticle{d}, Nothing)
smcioψ3 = SMCIO{modelψ.particle, modelψ.pScratch}(N ÷ 8, n, 1, true)

smc!(modelψ3, smcioψ3)

smcioψ3.logZhats[end] - truelogZ


map(s -> minimum(s.esses), [smcio, smcioψ, smcioψ2, smcioψ3])

map(s -> length(countmap(s.eves)), [smcio, smcioψ, smcioψ2, smcioψ3])

map(s -> s.logZhats[end], [smcio, smcioψ, smcioψ2, smcioψ3]) .- truelogZ

map(s -> SequentialMonteCarlo.V(s, x -> 1, true, false, n), 
[smcio, smcioψ, smcioψ2, smcioψ3])


map(s -> minimum([mean(getfield.(s.allZetas[i], :β₁)) for i in 1:n]), [smcioψ, smcioψ2, smcioψ3])



Nmc= 8

DMβ = DecompTemperKernel{eltype(bestψ)}(log(0.5), Nmc)
Dchainψ = DecompTwistedMarkovChain(μ, M, DMβ, n, bestψ2, Nmc)

Dpotentialψ = MCDecompTwistedLogPotentials(potential)

Dmodelψ = SMCModel(Dchainψ, Dpotentialψ, n, DecompTwistVectorParticle{d}, Nothing)

Dsmcioψ = SMCIO{Dmodelψ.particle, Dmodelψ.pScratch}(N*100, n, 1, true)

smc!(Dmodelψ, Dsmcioψ)

Dsmcioψ.logZhats[end] .- truelogZ


map(s -> minimum(s.esses), [smcio, smcioψ, smcioψ2, smcioψ3, Dsmcioψ])
map(s -> s.logZhats[end], [smcio, smcioψ, smcioψ2, smcioψ3, Dsmcioψ]) .- truelogZ

minimum([mean(getfield.(getfield.(Dsmcioψ.allZetas[i], :twₚ₊₁),:β)) for i in 1:n])


i = 20; map(s -> (s[i].J \ s[i].h, s[i].J), [bestψ, bestψ2, bestψ3])


smc!(model, smcio); smcio.logZhats[end] - truelogZ


# update new version to handle just lambda = 1

# recreate PhD experiments