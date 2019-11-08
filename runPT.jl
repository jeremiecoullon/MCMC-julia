#!/usr/bin/env julia

using Distributed
@everywhere using Distributions
using Plots
@everywhere include("./src/MH.jl")
include("./src/PT.jl")


@everywhere MoG = MixtureModel(Normal[
   Normal(40.0, 0.8),
   Normal(-40., 0.8),
        ])

@everywhere function logPostTemp(x::Float64, ß::Float64)::Float64
    sleep(1)
    logpdf(MoG, x)*ß
end


# =========
# Define RE sampler
chain1 = TempChain(MH.Chain(0,0, [3.3], 0), 3)
chain2 = TempChain(MH.Chain(0,0, [3.], 0), 13)
chain3 = TempChain(MH.Chain(0,0, [3.], 0), 50)
chain4 = TempChain(MH.Chain(0,0, [3.], 0), 40)
chain5 = TempChain(MH.Chain(0,0, [3.], 0), 30)

ensemble = initializeEnsembleLogPostCurrent(Ensemble(
    Dict(1 => chain1, 0.1 => chain2, 0.01 => chain3, 0.03 => chain4, 0.05 => chain5),
))

N = 20

# Replica Exchange sampler
@time begin
for i in range(0, stop=N)
    withinTemperatureMove(ensemble, logPostTemp)
    betweenTemperatureMove(ensemble, logPostTemp)
end
end



println("Acceptance rates: ", [MH.acceptanceRate(v.chain) for (k,v) in ensemble.chains])


# Animate traceplots and histogram
# myarray = zeros(0)
# anim = @animate for i in ensemble.chains[1].chain.samples[1:100:end]
#     push!(myarray, i)
#     # plot(myarray)
#     plot(histogram(myarray, bins=200, label="PT"),
#         plot(myarray, label="PT"),
#         layout=(2,1))
# end
# gif(anim, "animations/PTsampler-test.gif", fps = 30)
