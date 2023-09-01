using SequentialMonteCarlo
using Distributions
using LinearAlgebra
using StaticArrays
using Random
import StatsBase: countmap
using Kalman
using GaussianDistributions
import GaussianDistributions: ⊕
using PathTwister

const MAXITER = 10^5

N = 2^11
Nmc = 2
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

function ⊗(x::Gaussian, y::Gaussian)
    Σ⁻¹ = inv(x.Σ) + inv(y.Σ)
    μ = Σ⁻¹ \ ((x.Σ \ x.μ) + (y.Σ \ y.μ))
    Gaussian(μ, inv(Σ⁻¹))
end

function ⨸(x::Gaussian, y::Gaussian)
    Σ⁻¹ = inv(x.Σ) - inv(y.Σ)
    μ = Σ⁻¹ \ ((x.Σ \ x.μ) - (y.Σ \ y.μ))
    Gaussian(μ, inv(Σ⁻¹))
end

## Kalman Filter
D = Matrix(Diagonal(ones((d,))))  

# Define evolution scheme
Kevo = LinearEvolution(A, Gaussian(b, Σ))

# Define observation scheme
Obs = LinearObservationModel(D, D)
Mkf = LinearStateSpaceModel(Kevo, Obs)

Y = collect(t=>y[t] for t in 1:n)
Mkf0 = Gaussian(zeros(d), Matrix(μ.Σ))

# Filter
filtering, truelogZ = kalmanfilter(Mkf, 1 => Mkf0, Y)
smoothing, _ = rts_smoother(Mkf, 1 => Mkf0, Y)

Gpot = Gaussian.(y, [D])

optimaltwist = (Gpot .⊗ smoothing.x) .⨸ filtering.x

optψ = [ExpQuadTwist(g.Σ \ g.μ, inv(Matrix(g.Σ))) for g in optimaltwist]

## Base SMC

potential = MvNormalNoise(y, 1.0*I)

bmodel = SMCModel(chain, potential, n, VectorParticle{d}, Nothing)

bsmcio = SMCIO{bmodel.particle, bmodel.pScratch}(N, n, 1, true, κ)

smc!(bmodel, bsmcio)

logZ(bsmcio) - truelogZ

## Twisted

struct LearnSettings
    MCreps::Int64
    cvstrategy::Int64
    iter::Int64
    α::Float64
end

struct RepSettings
    reps::Int64
    targetα::Float64
    Nacc::Int64
    Nmc::Int64
end

function smctwist!(basemod::SMCModel, baseio::SMCIO, twistio::SMCIO, twistG::MCDecompTwistedLogPotentials, repset::RepSettings, learn::LearnSettings)

    compcost = 0.0

    smc!(basemod, baseio)
    compcost += baseio.N
    
    ψ = [ExpQuadTwist{Float64}(d) for _ in 1:n]
    lassocvtwist!(ψ, baseio, basemod, learn.MCreps, cvstrategy = learn.cvstrategy, iter = learn.iter, netα = learn.α)

    for i in 1:repset.reps
        βkernel = DecompTemperKernel{eltype(ψ)}(log(repset.targetα), repset.Nacc)
        twistchain = DecompTwistedMarkovChain(μ, M, βkernel, n, ψ, repset.Nmc)
        twistmodel = SMCModel(twistchain, twistG, n, DecompTwistVectorParticle{d}, DecompTwistedScratch{d, eltype(ψ)})
        
        smc!(twistmodel, twistio)
        rjcost = mean(mean.([getfield.(particles, :rsn) for particles in twistio.allZetas]))
        compcost += twistio.N * (rjcost + βkernel.Nₐ + twistchain.Nₘ)

        if i < repset.reps
            lassocvtwist!(ψ, twistio, twistmodel, learn.MCreps, cvstrategy = learn.cvstrategy, iter = learn.iter, netα = learn.α)
        end
    end

    return compcost

end

learncv = LearnSettings(1, 8, 8, 1.0)
repset = RepSettings(3, 0.005, 1, 1)
tpotential = MCDecompTwistedLogPotentials(potential)
tsmcio = SMCIO{DecompTwistVectorParticle{d}, DecompTwistedScratch{d, ExpQuadTwist{Float64}}}(N, n, 1, true, κ)
smctwist!(bmodel, bsmcio, tsmcio, tpotential, repset, learncv)
logZ(tsmcio) - truelogZ

# try d = 3, long time series (paper examples)

# bring tsmcmodel out, to run again smc!(...)
# increase particles and decrease target acceptance rate
# see what's taking so long (profile) - bad estimation of rejection part - long times

ψ1 = [ExpQuadTwist{Float64}(d) for _ in 1:n]
lassocvtwist!(ψ1, bsmcio, bmodel, 8, cvstrategy = 8, iter = 4, netα = 1.0)

Mβ₀ = TemperKernel{eltype(ψ1)}(log(0.25), Nmc)
chainψ₀ = DecompTwistedMarkovChain(μ, M, Mβ₀, n, ψ1, Nmc)
modelψ₀ = SMCModel(chainψ₀, potentialψ, n, DecompTwistVectorParticle{d}, DecompTwistedScratch{d, eltype(ψ1)})
smcioψ₀ = SMCIO{modelψ₀.particle, modelψ₀.pScratch}(N, n, 1, true, κ)

smc!(modelψ₀, smcioψ₀)
logZ(smcioψ₀) - truelogZ

Mβ₁ = DecompTemperKernel{eltype(ψ1)}(log(0.5), Nmc)
chainψ₁ = DecompTwistedMarkovChain(μ, M, Mβ₁, n, ψ1, Nmc)
modelψ₁ = SMCModel(chainψ₁, potentialψ, n, DecompTwistVectorParticle{d}, DecompTwistedScratch{d, eltype(ψ1)})

smcioψ₁ = SMCIO{modelψ₁.particle, modelψ₁.pScratch}(N, n, 1, true, κ)
smc!(modelψ₁, smcioψ₁)
logZ(smcioψ₁) - truelogZ

mean.([getfield.(particles, :rsn) for particles in smcioψ₁.allZetas])

ψ2 = deepcopy(ψ1)
lassocvtwist!(ψ2, smcioψ₁, modelψ₁, 8, cvstrategy = 8, iter = 4, netα = 1.0)

Mβ₂ = DecompTemperKernel{eltype(ψ2)}(log(0.05), Nmc)
chainψ₂ = DecompTwistedMarkovChain(μ, M, Mβ₂, n, ψ1, Nmc)
modelψ₂ = SMCModel(chainψ₂, potentialψ, n, DecompTwistVectorParticle{d}, DecompTwistedScratch{d, eltype(ψ2)})

smcioψ₂ = SMCIO{modelψ₂.particle, modelψ₂.pScratch}(N, n, 1, true, κ)
smc!(modelψ₂, smcioψ₂)
logZ(smcioψ₂) - truelogZ

mean.([getfield.(particles, :rsn) for particles in smcioψ₂.allZetas])


ψ3 = deepcopy(ψ2)
lassocvtwist!(ψ3, smcioψ₂, modelψ₂, 8, cvstrategy = 8, iter = 4, netα = 1.0)

Mβ₃ = DecompTemperKernel{eltype(ψ2)}(log(0.001), Nmc)
chainψ₃ = DecompTwistedMarkovChain(μ, M, Mβ₃, n, ψ1, Nmc)
modelψ₃ = SMCModel(chainψ₃, potentialψ, n, DecompTwistVectorParticle{d}, DecompTwistedScratch{d, eltype(ψ3)})

smcioψ₃ = SMCIO{modelψ₃.particle, modelψ₃.pScratch}(N, n, 1, true, κ)
smc!(modelψ₃, smcioψ₃)
logZ(smcioψ₃) - truelogZ

mean.([getfield.(particles, :rsn) for particles in smcioψ₃.allZetas])

# comparison SMC

rsiters = [getfield.(particles, :rsn) for particles in smcioψ₃.allZetas]
multi = mean(mean.(rsiters))

Neq = ceil(Int64, N * (multi + Nmc))

bsmcio = SMCIO{bmodel.particle, bmodel.pScratch}(N*4, n, 1, true, κ)

baseZ = zeros(100)

for i in 1:100
    smc!(bmodel, bsmcio)
    baseZ[i] = logZ(bsmcio) - truelogZ
end

ψZ = zeros(100)
for i in 1:100
    smc!(modelψ₀, smcioψ₀)
    ψZ[i] = logZ(smcioψ₀) - truelogZ
end

ψZ₃ = zeros(100)
for i in 1:100
    smc!(modelψ₃, smcioψ₃)
    ψZ₃[i] = logZ(smcioψ₃) - truelogZ
end

mean(ψZ₃)
mean(ψZ)
mean(baseZ)



# Does the tempering of ψ affect the optimal tempering?
# ψ = λ * (ψ / λ)ᵝ = ψᵝ * λ¹⁻ᵝ <--- bad?
# ψ = λ * (ψᵝ / λ) = ψᵝ

# Can we reduce variance of random weights by using
# new λ just for MC estimate?



N = 2^12
potentialψopt = MCDecompTwistedLogPotentials(potential)
Mβopt = TemperKernel{eltype(optψ)}(log(0.02), 1)
chainψopt = DecompTwistedMarkovChain(μ, M, Mβopt, n, optψ, 1)
modelψopt = SMCModel(chainψopt, potentialψopt, n, DecompTwistVectorParticle{d}, DecompTwistedScratch{d, eltype(optψ)})
smcioψopt = SMCIO{modelψopt.particle, modelψopt.pScratch}(N, n, 1, true, κ)

smc!(modelψopt, smcioψopt)
logZ(smcioψopt) - truelogZ

rjcost = mean(mean.([getfield.(particles, :rsn) for particles in smcioψopt.allZetas]))

Neq = ceil(Int64, N*(rjcost+3))

bsmcio = SMCIO{bmodel.particle, bmodel.pScratch}(Neq, n, 1, true, κ)

smc!(bmodel, bsmcio)
logZ(bsmcio) - truelogZ