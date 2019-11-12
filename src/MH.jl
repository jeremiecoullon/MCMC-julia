
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

# =====
# `getLastSample` and `appendNewSample` deal with 1D and multivariate distributions

function getLastSample(samplesArray::Array{Float64, 2})
    samplesArray[:, end]
end

function getLastSample(samplesArray::Array{Float64, 1})
    samplesArray[end]
end


function appendNewSample(samplesArray::Array{Float64, 2}, xNew::Array{Float64, 1})
    hcat(samplesArray, xNew)
end

function appendNewSample(samplesArray::Array{Float64, 1}, xNew::Float64)
    push!(samplesArray, xNew)
end

# =====

function initializeLogPostCurrent(samples::Chain, logPost::Function)
    samples.logPostCurrent = logPost(getLastSample(samples.samples))
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

function getNextSample(sample, logPostCurrent::Float64, logPost::Function, proposalDist)
    xNew = sample + rand(proposalDist)
    logPostNew = logPost(xNew)
    acceptBool = acceptReject(logPostNew, logPostCurrent)
    acceptBool ? (xNew, acceptBool, logPostNew) : (sample, acceptBool, logPostCurrent)
end

function metropolisHastingsStep(samples::Chain, logPost::Function, proposalDist)
    xNew, acceptBool, logPostNew = getNextSample(getLastSample(samples.samples),
                                    samples.logPostCurrent,
                                    logPost,
                                    proposalDist)
    # push!(samples.samples, xNew)
#     samples.samples = hcat(samples.samples, xNew)
    samples.samples = appendNewSample(samples.samples, xNew)
    if acceptBool
        samples.accept += 1
        samples.logPostCurrent = logPostNew
    else
        samples.reject += 1
    end
    samples
end

end
