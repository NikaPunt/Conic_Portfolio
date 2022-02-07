using Polynomials
include("implied_moments.jl")

Base.@kwdef struct VG
    σ::Float64
    ν::Float64
    θ::Float64
    Δt::Float64=1
end

#getVGParams(v,s,k,T,Tnew) returns the struct VG that has the VG(σ,ν,θ) parameterization given 
#the variance v, skewness s and kurtosis k of the returns X_T where T is a specific
#horizon.
#
#Input:     v
#           Type:           Float64
#           Description:    Variance of returns.
#
#           s
#           Type:           Float64
#           Description:    Skewness of returns.
# 
#           k
#           Type:           Float64
#           Description:    Kurtosis of returns.
#           
#           T
#           Type:           Number
#           Description:    Time horizon of the timeseries. Per standard this is set to 1.
#
#           Tnew
#           Type:           Number
#           Description:    A new time horizon to return all VG parameters so that the VG process
#                           models in the new time horizon. Suppose X_T were the daily returns, then
#                           T=1 means a time horizon of one, and Tnew=20 is a time horizon of a month.
#                           Per standard Tnew is set to 1.
#
#Output:    VG(σ,ν,θ)
#           Type:           VG
#           Description:    A three-parameter struct that contains the parameterization of a VG distribution.
#
function getVGParams(v::Float64,s::Float64,k::Float64,T::Number=1,Tnew::Number=1)
    if 6+3*s^2-2*k < 0
        c_1 = ((k/3)-1)*T/(v*s^2)
        c_2 = 3*((k/3-1)/s^2)-1
        c_3 = 2*(v^2/T^2)*(1-2*((k/3-1)/s^2))
        σ_sq = calculateσ(c_1,c_2,c_3)
        # print("sigma squared: ",σ_sq,"\n")
    else
        s_list = calculateS(9,(20-6*k),-2*s)
        s_new = s_list[argmin(abs.(s_list .- s))]
        k_new = 3+1.5*s_new^2+0.01
        c_1 = ((k_new/3)-1)*T/(v*s_new^2)
        c_2 = 3*((k_new/3-1)/s_new^2)-1
        c_3 = 2*(v^2/T^2)*(1-2*((k_new/3-1)/s_new^2))
        σ_sq = calculateσ(c_1,c_2,c_3)
        # print("sigma squared: ",σ_sq,"\n")
    end
    ν = (s^2*v^3/(T^2*(σ_sq+2*v/T)^2*(v/T-σ_sq)))
    # print((v/T-σ_sq)/ν,"\n\n v: ",v,"\n\n σ_sq: ",σ_sq,"\n\n")
    θ = sign(s)*√((v/T-σ_sq)/ν)
    σ = √(σ_sq)
    # print("sigma: ",σ,"\n")
    return VG(σ=σ,ν=ν,θ=θ,Δt=Tnew)
end

function getVGParams(impliedparams,T=1,Tnew=1)
    v = impliedparams.v
    s = impliedparams.s
    k = impliedparams.k
    return getVGParams(v,s,k,T,Tnew)
end

function calculateσ(c_1,c_2,c_3)
    R = -(c_3)/(2*c_1)-(c_2/(3*c_1))^3
    Q = -(c_2^2)/(9*c_1^2)
    D = ComplexF64(Q^3 + R^2)
    x1 = -(c_2/(3*c_1))+(R+√(D))^(1/3)+(R-√(D))^(1/3)
    return Float64(x1)
end

function calculateS(a,c,d)
    Q = 3*a*c/(9*a^2)
    R = -a^2*d/(2*a^3)
    D = ComplexF64(Q^3+R^2)
    S = (R+√(D))^(1/3)
    T = (R-√(D))^(1/3)
    realRoots = Vector{Float64}([Inf64,Inf64,Inf64])
    if iszero(imag(S+T))
        realRoots[1] = Float64(S+T)
    end
    if iszero(imag(-(S+T)/2 + 1im*√(3)*(S-T)/2))
        realRoots[2] = Float64(-(S+T)/2 + 1im*√(3)*(S-T)/2)
    end
    if iszero(imag(S+T))
        realRoots[3] = Float64(-(S+T)/2 - 1im*√(3)*(S-T)/2)
    end
    return realRoots
end

