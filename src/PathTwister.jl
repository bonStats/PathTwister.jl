module PathTwister

using SequentialMonteCarlo
using GLMNet

import Random: GLOBAL_RNG
import Distributions: MvNormalCanon, mode, logpdf
import LinearAlgebra: det, diag, diagind

include("abstract.jl")
export AbstractParticle, AbstractTwist, ParameterizedDistribution, value, AbstractMarkovChain

include("markov-chain.jl")
export MarkovChain, HomogeneousMarkovChain

include("exp-quad-twist/exp-quad-struct.jl")
include("exp-quad-twist/exp-quad-learn.jl")
export ExpQuadTwist, lassocvtwist!

end
