using Polynomials
include("implied_moments.jl")

struct VG
    σ::Float64
    ν::Float64
    θ::Float64
end

function getVGParams(impliedparams,T)
    v = impliedparams.v
    s = impliedparams.s
    k = impliedparams.k
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
    ν = (s^2*v^3/(T^2*(σ_sq+2*v/T)^2(v/T-σ_sq)))
    # print((v/T-σ_sq)/ν,"\n\n v: ",v,"\n\n σ_sq: ",σ_sq,"\n\n")
    θ = sign(s)*√((v/T-σ_sq)/ν)
    σ = √(σ_sq)
    # print("sigma: ",σ,"\n")
    return VG(σ,ν,θ)
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