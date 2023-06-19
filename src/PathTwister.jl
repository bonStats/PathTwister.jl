module PathTwister

using SequentialMonteCarlo
using GLMNet

import Random: GLOBAL_RNG, AbstractRNG
import Distributions: Sampleable
import LinearAlgebra: det, diag, diagind, Diagonal, I, isposdef, Symmetric, Eigen, eigen, eigen!, dot
import StatsFuns: logsumexp
import Roots: find_zero

include("abstract.jl")
export AbstractParticle, AbstractTwist, MarkovKernel, AbstractMarkovChain, LogPotentials
export value, untwist

include("markov-chain.jl")
export MarkovChain, HomogeneousMarkovChain

include("exp-quad-twist/mvcanon-glmnet-cv.jl")
include("exp-quad-twist/exp-quad-struct.jl")
include("exp-quad-twist/exp-quad-learn.jl")
export ExpQuadTwist, lassocvtwist!

include("rejection-sampler.jl")
export RejectionSampler, rand, rand!

include("adaptive-rejection-dist.jl")
export AdaptiveRejection

include("flat-twist.jl/flat-struct.jl")

end
