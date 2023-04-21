struct AdaptiveRejection{S<:Sampleable,T<:AbstractTwist}
    d::S
    ψ::T
    logα::Float64 # acceptance target
    Nₐ::Int64 # sample to estimate acceptance rate
    MAXITER::Int64
end

function (tw::AdaptiveRejection)()
    # choose β from trial draws
    logψ = tw.ψ.(rand(tw.d, tw.Nₐ), :log)

    # define reach_acc_rate(b) > 0 if accept target is exceeded
    reach_acc_rate(b::Float64) = logsumexp(b .* logψ) - log(tw.Nₐ) - tw.logα
    if reach_acc_rate(1.0) > 0
        β = 1.0
    else
        β = find_zero(reach_acc_rate, (0,1))
    end

    return RejectionSampler(tw.d, tw.ψ, β, tw.MAXITER)

end