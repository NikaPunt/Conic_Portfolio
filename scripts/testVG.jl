include("getVGParams.jl")
using Distributions
# using LinearAlgebra

randpars = [rand(3)*2 ; 1]
testDist = VG(randpars...)
σ_m = testDist.σ
ν_m = testDist.ν
θ_m = testDist.θ
# This model carries within the variance,
# skewness and kurtosis. Given by:
v_m = σ_m^2+ν_m*θ_m^2
s_m = (θ_m*ν_m*(3*σ_m^2+2*ν_m*θ_m^2))/v_m^(3/2)
k_m = 3*(1+2*ν_m-(ν_m*σ_m^4)/v_m^2)

getVGParams(v_m,s_m,k_m)

function VG2MRet(params::VG,M::Integer)
    returns = zeros(M)
    Δt = params.Δt
    σ = params.σ
    ν = params.ν
    θ = params.θ
    g = Gamma(Δt*1/ν,ν)
    g_n = rand(g,M)
    ϵ = Normal()
    ϵ_n = rand(ϵ,M)
    for j = 1:M
        returns[j] = θ*(g_n[j])+σ*√(g_n[j])*ϵ_n[j]
    end
    return returns
end

Returns = VG2MRet(testDist,1000000)
v = var(Returns)
s = skewness(Returns)
k = kurtosis(Returns)+3

## Let us test monthly returns
# daily σ, ν, θ through rand(3)*2 and month = 20 days
randpars = [rand(3)*2 ; 20]
testDist = VG(randpars...)

monthReturns = VG2MRet(testDist,1000000)
v = var(monthReturns)
s = skewness(monthReturns)
k = kurtosis(monthReturns)+3
# we expect that v s k give us √(Δt)⋅σ, ν/Δt, θ⋅Δt
VGNew = getVGParams(v,s,k)
testDist