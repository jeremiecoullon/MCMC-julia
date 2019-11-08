#!/usr/bin/env julia

using Distributed
using Distributions
using Plots

include("./src/MH.jl")

# ==================


mu = 2
sd = 1
function logPost(x::Float64)::Float64
    -(0.5/(sd^2))*(x-mu)^2
end
# Metropolis Sampler
N = 6000
sdProp = 3
gaussianProposal = Normal(0, sdProp)
samples = MH.initializeLogPostCurrent(MH.Chain(0, 0, [0.], 0), logPost)

for i in range(0, stop=N)
    MH.metropolisHastingsStep(samples, logPost, gaussianProposal)
end

ar = MH.acceptanceRate(samples)
println("Acceptance rate: $ar%")
postmean, postsd = round(mean(samples.samples), digits=2), round(std(samples.samples), digits=2)
println("Posterior mean and sd: $postmean, $postsd")
println("True mean and sd: $mu, $sd")
# plot(samples.samples, show=true)
# ==================

# Animate
# myarray = zeros(0)
# anim = @animate for i in samples.samples[1:20:end]
#     push!(myarray, i)
#     plot(histogram(myarray, bins=100, label="MH"),
#         plot(myarray, label="MH"),
#         layout=(2,1))
# end
# gif(anim, "animations/MH_sampler.gif", fps = 30)
