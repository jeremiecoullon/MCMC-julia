using Distributed
using Distributions
include("./MH.jl")

# To make parallel:
# have both maps be 'pmaps'
# add @everywhere to doWorkNextSample(x) (in this file)
# in script: add @everywhere to "using Distributions", "include("./src/MH.jl")", "logPostTemp"

mutable struct TempChain
    chain::MH.Chain
    propSd::Float64
end

mutable struct Ensemble
    chains::Dict{Float64, TempChain}
end


function initializeEnsembleLogPostCurrent(ensemble::Ensemble)
   for (k,v) in ensemble.chains
        v.chain.logPostCurrent = logPostTemp(v.chain.samples[end], k)
    end
    ensemble
end


function withinTemperatureMove(ensemble::Ensemble, logPost::Function)
    """
    Within-temperature move: Gaussian proposal
    """
    arrayMapParams = [(k,
                        v.chain.samples[end],
                        v.chain.logPostCurrent,
                        x -> logPost(x,k),
                        Normal(0, v.propSd))
                    for (k,v) in ensemble.chains]
    @everywhere function doWorkNextSample(x)
        (x[1], MH.getNextSample(x[2:5]...)...)
    end
    # get the updated parameters for each chain
    arrayNextSamples = pmap(doWorkNextSample, arrayMapParams)

    # append the updated parameters to each chain
    for x in arrayNextSamples
        k, xNew, acceptBool, logPostNew = x
        push!(ensemble.chains[k].chain.samples, xNew)
        acceptBool ? (ensemble.chains[k].chain.accept += 1) : (ensemble.chains[k].chain.reject += 1)
        ensemble.chains[k].chain.logPostCurrent = logPostNew
    end
end


function choosePair(ensemble::Ensemble)
    tempArray = sort([k for k in keys(ensemble.chains)])
    pairsArray = [[tempArray[i-1], tempArray[i]] for i in 2:length(tempArray)]
    rand(pairsArray)
end


function betweenTemperatureMove(ensemble::Ensemble, logPost)
    temp1, temp2 = choosePair(ensemble)

    sample1 = ensemble.chains[temp1].chain.samples[end]
    sample2 = ensemble.chains[temp2].chain.samples[end]

    function doWorkLogPost(x)
        logPost(x[1], x[2])
    end
    logPostNew1, logPostNew2 = pmap(doWorkLogPost, [(sample1, temp2), (sample2, temp1)])
    # logPostNew1, logPostNew2 = logPost(sample1, temp2), logPost(sample2, temp1)
    logPostNew = logPostNew1 + logPostNew2
    logPostCurrent = ensemble.chains[temp1].chain.logPostCurrent + ensemble.chains[temp2].chain.logPostCurrent

    acceptBool = MH.acceptReject(logPostNew, logPostCurrent)
    if acceptBool
        push!(ensemble.chains[temp1].chain.samples, sample2)
        push!(ensemble.chains[temp2].chain.samples, sample1)
        ensemble.chains[temp1].chain.accept += 1
        ensemble.chains[temp2].chain.accept += 1
        for k in filter(x -> !(x in [temp1, temp2]), keys(ensemble.chains))
            # update all other chains
            push!(ensemble.chains[k].chain.samples, ensemble.chains[k].chain.samples[end])
            ensemble.chains[k].chain.reject += 1
        end
        ensemble.chains[temp1].chain.logPostCurrent = logPostNew2
        ensemble.chains[temp2].chain.logPostCurrent = logPostNew1
    else
        for k in keys(ensemble.chains)
            ensemble.chains[k].chain.reject += 1
            push!(ensemble.chains[k].chain.samples, ensemble.chains[k].chain.samples[end])
        end
        ensemble.chains[temp1].chain.reject += 1
        ensemble.chains[temp2].chain.reject += 1
    end
    ensemble
end
