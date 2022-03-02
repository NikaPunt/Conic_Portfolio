include("getVGParams.jl")
# Given a vector of VG parameters, we create M returns for each entry in the vector.
function VG_Params2MRet(params::Vector{VG},M::Integer)
    N = length(params)
    returns = zeros(N,M)
    for i = 1:N
        σ = params[i].σ
        ν = params[i].ν
        θ = params[i].θ
        Δt = params[i].Δt
        g = Gamma(Δt*1/ν,ν)
        g_n = rand(g,M)
        ϵ = Normal()
        ϵ_n = rand(ϵ,M)
        for j = 1:M
            returns[i,j] = θ*(g_n[j])+σ*√(g_n[j])*ϵ_n[j]
        end
    end
    return returns
end