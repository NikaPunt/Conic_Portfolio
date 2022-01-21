# import Pkg
# Pkg.add("Distributions")
include("options.jl")
using Distributions

struct implied_params
    v::Float64
    s::Float64
    k::Float64
end

global function delK(i,Ks)
    N = length(Ks)
    if i==1
        return Ks[2]-Ks[1]
    elseif i==N
        return Ks[N]-Ks[N-1]
    else
        return (Ks[i+1]-Ks[i-1])/2
    end
end

function Price(calls,puts,K0,i)
    Ki = calls.strikes[i]
    if Ki < K0
        return puts.prices[i]
    elseif Ki == K0
        return (calls.prices[i]+puts.prices[i])/2
    else
        return calls.prices[i]
    end
end

function implied_moments(n,S0,T,q,r,calls::options,puts::options)
    if calls.type != call
        return error("Not a valid call list")
    end
    if puts.type != put
        return error("Not a valid put list")
    end
    F0 = S0*exp((r-q)*T)
    K0 = maximum(calls.strikes[calls.strikes .< F0])
    A = log(K0/S0)
    B = (F0/K0)-1
    bigsum = 0
    M = length(calls.strikes)
    for i = 1:M
        dK = delK(i,calls.strikes)
        K = calls.strikes[i]
        sumA = log(K/S0)
        bigsum = bigsum + dK/K^2*((n-1)*sumA^(n-2)-sumA^(n-1))*
            Price(calls,puts,K0,i)
    end
    return A^n + n*A^(n-1)*B + exp(r*T)*n*bigsum
end



function get_params(S0,T,q,r,calls::options,puts::options)
    one = implied_moments(1,S0,T,q,r,calls::options,puts::options)
    two = implied_moments(2,S0,T,q,r,calls::options,puts::options)
    three = implied_moments(3,S0,T,q,r,calls::options,puts::options)
    four = implied_moments(4,S0,T,q,r,calls::options,puts::options)
    v = two - one^2
    s = (three -3*one*two + 2*one^3)/(v)^(3/2)
    k = (four - 4*one*three+6*one^2*two-3*one^4)/v^2
    return implied_params(v,s,k)
end


#get_params_timeseries_returns returns the daily variance v, skewness s and kurtosis k
#from a Float64 matrix of daily returns.
#
#Input:     returnsMatrix
#           Type:           Matrix{Float64}
#           Description:    An n×m matrix with a timeseries of returns in each row.
#                           The 2×7 matrix 
#                           [[.1, .2, .3, .4, .5, .6, .7],
#                           [.1, .2, .4, .5, .6, .3, .4]] 
#                           has two timeseries, each with their own v,s,k matrix.
#
#Output:    vskMatrix
#           Type:           Matrix{Float64}
#           Description:    An n×3 matrix where each row coincides with the timeseries 
#                           in returnsMatrix. Column 1, 2 and 3 contain the timeseries' v, s and k respectively.
#
#Note:      If you are attempting to get (v,s,k) for an independent component, remember that
#           an ICA cannot uniquely define the variance of the component.
#           That is why v is always ≈1.0
function get_params_timeseries_returns(returnsMatrix::Matrix{Float64})
    n = size(returnsMatrix,1)
    vskMatrix = zeros(n,3)
    for i=1:n
        ts = returnsMatrix[i,:]
        v,s,k = (var(ts),skewness(ts),kurtosis(ts)+3)
        vskMatrix[i,:] = [v,s,k]
    end
    return vskMatrix
end