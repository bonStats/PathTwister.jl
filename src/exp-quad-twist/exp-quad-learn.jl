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
        # single design matrix (linear only)
        X_single = value(smcio.allZetas[p])'

        if p == model.maxn
            # extra particles for same number of observations over p
            if MCreps > 1
                padparticles = [[model.particle() for _ in 1:smcio.N] for _ in 2:MCreps]
                pushfirst!(padparticles, smcio.allZetas[model.maxn])
                for t in 2:MCreps
                    untwist(model.M!).(padparticles[t], [GLOBAL_RNG], [model.maxn], smcio.allZetas[model.maxn-1], [nothing]) #rng
                end
            else 
                padparticles = [smcio.allZetas[model.maxn]] # ensurealways a vector (of Vector{Particle})
            end

            y = vcat([buildresponse(padparticles[i], model, model.maxn) for i in 1:MCreps]...)
            Xlin = vcat(X_single, [value(padparticles[i])' for i in 2:MCreps]...)
            
            # # adjust regression for current ψ
            # if !iszero(ψs[model.maxn])
            #     y .-= vcat([ψs[model.maxn](padparticles[i], :log) for i in 1:MCreps]...)
            # end

        else # p < model.maxn
            y = vcat([buildresponse(smcio.allZetas[p], model, p, ψs[p+1]) for i in 1:MCreps]...) # automatically adjusts for old+new ψ by updating ψs below
            #X = repeat(X_single, MCreps) # space efficient version?
            Xlin = @view X_single[repeat(1:end,MCreps),:]

            # # adjust regression for current ψ
            # if !iszero(ψs[p])
            #     y .-= repeat(ψs[p](smcio.allZetas[p], :log), MCreps)
            # end
        end

        if iszero(ψs[p])
            envm = learn_mvcanon_cvnet(Xlin, y, folds, 2; alpha = 1.0)
        else
            envm = EigenMvNormalCanon(ψs[p])
            learn_mvcanon_cvnet!(envm, Xlin, y, folds, 2; alpha = 1.0)
        end

        ψs[p] = ExpQuadTwist(envm)

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

