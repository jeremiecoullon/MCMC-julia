
module MH
# Some functions for Metropolis Hastings MCMC


using Distributions
export Chain, initializeLogPostCurrent, acceptanceRate, acceptReject, metropolisHastingsStep, getNextSample

mutable struct Chain
    accept::Int
    reject::Int
    samples::Array{Float64}
    logPostCurrent::Float64
end

function initializeLogPostCurrent(samples::Chain, logPost)
    samples.logPostCurrent = logPost(samples.samples[end])
    samples
end

function acceptanceRate(samples::Chain)
    a = samples.accept
    r = samples.reject
    round(a / (a+r), digits=3)*100
end

function acceptReject(logPostNew::Float64, logPostCurrent::Float64)
    alphaRatio = logPostNew - logPostCurrent
    expSample = - rand(Exponential())
    if alphaRatio > expSample
        true
    else
        false
    end
end

function getNextSample(sample::Float64, logPostCurrent::Float64, logPost::Function, proposalDist)
    xNew = sample + rand(proposalDist)
    logPostNew = logPost(xNew)
    acceptBool = acceptReject(logPostNew, logPostCurrent)
    acceptBool ? (xNew, acceptBool, logPostNew) : (sample, acceptBool, logPostCurrent)
end

function metropolisHastingsStep(samples::Chain, logPost, proposalDist)
    xNew, acceptBool, logPostNew = getNextSample(samples.samples[end],
                                    samples.logPostCurrent,
                                    logPost,
                                    proposalDist)
    push!(samples.samples, xNew)
    if acceptBool
        samples.accept += 1
        samples.logPostCurrent = logPostNew
    else
        samples.reject += 1
    end
    samples
end

end
