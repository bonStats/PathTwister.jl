module PathTwister

using SequentialMonteCarlo
using GLMNet

import Random: GLOBAL_RNG, AbstractRNG

import LinearAlgebra: det, diag, diagind

include("abstract.jl")
export AbstractParticle, AbstractTwist, MarkovKernel, AbstractMarkovChain, LogPotentials
export value

include("markov-chain.jl")
export MarkovChain, HomogeneousMarkovChain

include("exp-quad-twist/exp-quad-struct.jl")
include("exp-quad-twist/exp-quad-learn.jl")
export ExpQuadTwist, lassocvtwist!

include("rejection-sampler.jl")
export samplebyrejection!

end
