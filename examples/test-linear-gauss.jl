using SequentialMonteCarlo
using Distributions
using LinearAlgebra
using StaticArrays
using Random
import StatsBase: countmap
using Kalman
using GaussianDistributions
using PathTwister

const MAXITER = 10^5

N = 2^10
Nmc = 2^3
κ = 0.5

# Types: VectorParticle, LinearGaussMarkovKernel, MvNormalNoise
include("linear-gauss-hmm.jl")

# Types: ExpTilt, TwistDecomp, DecompTemperAdaptSampler, DecompTwistVectorParticle, DecompTemperKernel, 
# DecompTwistedMarkovChain, MCDecompTwistedLogPotentials
#include("adaptive-temp-twist-partial.jl")
include("scratch-adaptive-temp-twist-partial.jl")

# helper
logZ(io::SMCIO) = io.logZhats[end]

# setup problem
n = 10
d = 10
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

## Kalman Filter
D = Matrix(Diagonal(ones((d,))))  

# Define evolution scheme
Kevo = LinearEvolution(A, Gaussian(b, Σ))

# Define observation scheme
Kobs = LinearObservation(Kevo, D, D)

# Filter
_, truelogZ = kalmanfilter(Kobs, 1 => Gaussian(zeros(d), Matrix(μ.Σ)), 1:n .=> y)

## Base SMC

potential = MvNormalNoise(y, 1.0*I)

bmodel = SMCModel(chain, potential, n, VectorParticle{d}, Nothing)

bsmcio = SMCIO{bmodel.particle, bmodel.pScratch}(N, n, 1, true, κ)

smc!(bmodel, bsmcio)

logZ(bsmcio) - truelogZ

## Twisted

potentialψ = MCDecompTwistedLogPotentials(potential)

ψ1 = [ExpQuadTwist{Float64}(d) for _ in 1:n]

lassocvtwist!(ψ1, bsmcio, bmodel, 8, cvstrategy = 8, iter = 4, netα = 1.0)

Mβ₀ = TemperKernel{eltype(ψ1)}(log(0.01), Nmc)
chainψ₀ = DecompTwistedMarkovChain(μ, M, Mβ₀, n, ψ1, Nmc)

modelψ₀ = SMCModel(chainψ₀, potentialψ, n, DecompTwistVectorParticle{d}, DecompTwistedScratch{d, eltype(ψ1)})

smcioψ₀ = SMCIO{modelψ₀.particle, modelψ₀.pScratch}(N, n, 1, true, κ)

smc!(modelψ₀, smcioψ₀)

logZ(smcioψ₀) - truelogZ

# comparison SMC

rsiters = [getfield.(particles, :rsn) for particles in smcioψ₀.allZetas]
multi = mean(mean.(rsiters))

Neq = ceil(Int64, N * (multi + Nmc))

bsmcio = SMCIO{bmodel.particle, bmodel.pScratch}(Neq, n, 1, true, κ)

smc!(bmodel, bsmcio)
logZ(bsmcio) - truelogZ