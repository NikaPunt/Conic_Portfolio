# cd("/home/nikap/Desktop/Masterthesis/Conic_Portfolio") # On Linux
using CSV
using DataFrames
using Statistics
using Plots
using IndependentComponentAnalysis
using Loess
using Distributions
# using Pkg; Pkg.add("BlackBoxOptim")
using(BlackBoxOptim)

include("ICA_assets.jl")
include("options.jl")
include("implied_moments.jl")


print(1)
csv_brk = CSV.File("BRK-B.csv");
# csv_brk_calls = CSV.File("brkb_calls.csv");
# csv_brk_puts = CSV.File("brkb_puts.csv");
csv_nke = CSV.File("NKE.csv");
# csv_nke_calls = CSV.File("nke_calls.csv");
# csv_nke_puts = CSV.File("nke_puts.csv");
csv_fb = CSV.File("FB.csv");
# csv_fb_calls = CSV.File("fb_calls.csv");
# csv_fb_puts = CSV.File("fb_puts.csv");
csv_v = CSV.File("V.csv");
# csv_v_calls = CSV.File("v_calls.csv");
# csv_v_puts = CSV.File("v_puts.csv");
csv_googl = CSV.File("GOOGL.csv");
# csv_googl_calls = CSV.File("googl_calls.csv");
# csv_googl_puts = CSV.File("googl_puts.csv");

df_brk = DataFrame(csv_brk) # NIKE(NKE) GOOGLE(GOOGL) VISA(V) META(FB) BERKSHIRE(BRK-A)
# df_brk_calls = DataFrame(csv_brk_calls)
# df_brk_puts = DataFrame(csv_brk_puts)
df_nke = DataFrame(csv_nke)
# df_nke_calls = DataFrame(csv_nke_calls)
# df_nke_puts = DataFrame(csv_nke_puts)
df_fb = DataFrame(csv_fb)
# df_fb_calls = DataFrame(csv_fb_calls)
# df_fb_puts = DataFrame(csv_fb_puts)
df_v = DataFrame(csv_v)
# df_v_calls = DataFrame(csv_v_calls)
# df_v_puts = DataFrame(csv_v_puts)
df_googl = DataFrame(csv_googl)
# df_googl_calls = DataFrame(csv_googl_calls)
# df_googl_puts = DataFrame(csv_googl_puts)

df_list = [df_brk,df_nke,df_fb,df_v,df_googl];
yields = [0,0.0078,0,0.0069,0];
# calls_list = df2option.([df_brk_calls,df_nke_calls,df_fb_calls,df_v_calls,df_googl_calls],call);
# puts_list = df2option.([df_brk_puts,df_nke_puts,df_fb_puts,df_v_puts,df_googl_puts],put);

nrAssets = length(df_list);
assetMeans = Vector{Float64}(undef,nrAssets);
assetShiftedReturns = Array{Float64,2}(undef,nrAssets,length(df_brk."Adj Close")-1);
for i = 1:nrAssets
    df = df_list[i]
    closes = df."Adj Close"
    returns = log.(closes[2:end]./closes[1:end-1])
    gemiddelde = mean(returns)
    assetMeans[i] = gemiddelde
    shiftedReturns = returns-repeat([gemiddelde],length(returns))
    assetShiftedReturns[i,:] = shiftedReturns;
end

print(2)
ica_comps = ICA_assets(assetShiftedReturns);

# X = A*S      where X are the returns, A is mixing matrix and S is independent comps
# S = W'*X
W = ica_comps.W;
A = ica_comps.mixing;
S = ica_comps.indcomps; #these are the independent components. For each row, we need σ, ν, θ. 

include("implied_moments.jl")
include("getVGParams.jl")
pars = get_params_timeseries_returns(S);

function VG_IndComp2MRet(params::Vector{VG},M::Integer)
    N = length(params)
    returns = zeros(N,M)
    for i = 1:N
        σ = params[i].σ
        ν = params[i].ν
        θ = params[i].θ
        g = Gamma(1/ν,1/ν)
        g_n = rand(g,M)
        ϵ = Normal()
        ϵ_n = rand(ϵ,M)
        for j = 1:M
            returns[i,j] = θ*(g_n[j]-1)+σ*√(g_n[j])*ϵ_n[j]
        end
    end
    return returns
end

# V = VG_IndComp2MRet([getVGParams(pars[1,:]...,1,20),
#                 getVGParams(pars[2,:]...,1,20),
#                 getVGParams(pars[3,:]...,1,20),
#                 getVGParams(pars[4,:]...,1,20),
#                 getVGParams(pars[5,:]...,1,20)], 10000)

print(3)
V = VG_IndComp2MRet([getVGParams(pars[1,:]...),
                getVGParams(pars[2,:]...),
                getVGParams(pars[3,:]...),
                getVGParams(pars[4,:]...),
                getVGParams(pars[5,:]...)], 10000)

#returns Y = A*V where A is the mixing matrix.
Y = A*V;
N,M = size(Y);
Returns = zeros(N,M);
for i = 1:N
    avgSumExp = 1/M*sum(exp.(Y[i,:]))
    # println(avgSumExp)
    for m = 1:M
        Returns[i,m] = exp(Y[i,m])-avgSumExp
    end
end

include("MINMAXVAR.jl")

# Returns for the ordered vector R the distorted expectation defined by distortion Ψ
# result == ψ(m/M) - Ψ((m-1)/M)    where m = 1,...,M and M is length(R)
function bid(R::Vector{Float64})
    ΨminusΨ = [MINMAXVAR(m/M,0.1)-MINMAXVAR((m-1)/M,0.1) for m=1:M]
    return ΨminusΨ' * R
end

# Reward μ_p = -∑ᵢ(aᵢ⋅bid(Rᵢ))
function REWARD(w::Vector{Float64},Rs::Matrix{Float64})
    μₚ = -w' * [    bid(sort(Rs[1,:])),
                    bid(sort(Rs[2,:])),
                    bid(sort(Rs[3,:])),
                    bid(sort(Rs[4,:])),
                    bid(sort(Rs[5,:]))]
    return μₚ
end

# Risk ̃c(a) = -bid(Rₚ)
function RISK(w::Vector{Float64},Rs::Matrix{Float64})
    Rₚ = Rs' * w
    c = -bid(sort(Rₚ))
    return c
end

function GAP(w::Vector{Float64},Rs::Matrix{Float64})
    c = RISK(w,Rs)
    μₚ = REWARD(w,Rs)
    return μₚ - c
end
print(4)

# # The following is some code using the julia package Optim. It does not converge very well.
# using Optim

# const BasePenalty = 10000 # Select some large number that will dominate the "normal" fitness.

# function penalty_constraint_sum_x_1(x)
#     if abs(sum(x)-1) > 0 
#         return BasePenalty + abs(sum(x)-1) # Smaller penalty as we get closer to non-violation of constraint
#     else
#         return 0.0
#     end
# end

# my_new_fitness(x) = -GAP(x,Returns) + penalty_constraint_sum_x_1(x)

# w_uni = repeat([1/N], N)
# lower = repeat([0],N)
# upper = repeat([1],N)
# result = optimize(my_new_fitness, lower, upper, rand(5), Fminbox(BFGS()))



# GAP(Optim.minimizer(result),Returns)

# function GAP(w::Vector{VariableRef},Rs::Matrix{Float64})
#     weightedReturns = transpose(w)*Rs
#     # bidPortfolio = bid(sort(weightedReturns'))
#     # bidIndividual = transpose(value.(w)) * [bid(sort(Rs[1,:])),
#     #     bid(sort(Rs[2,:])),
#     #     bid(sort(Rs[3,:])),
#     #     bid(sort(Rs[4,:])),
#     #     bid(sort(Rs[5,:]))]

#     # return bidPortfolio - bidIndividual
# end



# # lets start with the optimization calculations.
# # We need the following: 
# # maximize          bid(w'*Rs)-w'bid.(Rsorted)
# # subject to        sum(w) = 1
# #                   w⋅μ = μ_required
# #                   w > 0
# # where we assume all rows of Rsorted to be sorted.

# # let us discretize a bunch of values for μ_required
# μ = assetMeans;
# μ_required = range(minimum(assetMeans),stop=maximum(assetMeans),length=100);
# gap = zeros(100);


# # optimization
# model = Model();
# set_optimizer(model, Ipopt.Optimizer)
# @variable(model, w[1:N] >= 0) # you can unregister w through unregister(model, w)
# @constraint(model, lt1, sum(w) == 1);
# @objective(model, Max, GAP(w,Returns));
# optimize!(model);

const BasePenalty = 10000 # Select some large number that will dominate the "normal" fitness.

function penalty_constraint_sum_x_1(x)
    if abs(sum(x)-1) > 0 
        return BasePenalty + abs(sum(x)-1) # Smaller penalty as we get closer to non-violation of constraint
    else
        return 0.0
    end
end

function penalty_constraint_fixed_reward(x::Vector{Float64},Rs::Matrix{Float64},μₚ::Float64)
    if abs(REWARD(x,Rs)-μₚ) > 0 
        return BasePenalty + abs(REWARD(x,Rs)-μₚ)/(μₚ) # Smaller penalty as we get closer to non-violation of constraint
    else
        return 0.0
    end
end

I = 0.0247:0.001:0.0447
ws = zeros(length(I),N)

println("Begin optimization:")

for i = 1:length(I)
    println(i)
    μₚ = I[i]
    my_new_fitness(x) = RISK(x,Returns) + penalty_constraint_sum_x_1(x) + penalty_constraint_fixed_reward(x,Returns,μₚ)
    res = bboptimize(my_new_fitness; SearchRange=(0,1),NumDimensions=N,TraceMode=:silent)
    w_best = best_candidate(res)
    ϵᵣ = (REWARD(w_best,Returns)-μₚ)/μₚ
    if ϵᵣ < 0.0001
        ws[i,:] = w_best
    end
end

plot([RISK(ws[i,:],Returns) for i = 1:length(I)], [REWARD(ws[i,:],Returns) for i = 1:length(I)],seriestype = :scatter)

CSV.write("OPT_w.txt", DataFrame(ws,:auto),header=false);
CSV.write("OPT_Y.txt", DataFrame(Y:auto),header=false);
CSV.write("OPT_A.txt", DataFrame(A:auto),header=false);
CSV.write("OPT_V.txt", DataFrame(V:auto),header=false);
CSV.write("OPT_RETURNS.txt", DataFrame(Returns:auto),header=false);

my_new_fitness(x) = RISK(x,Returns) + penalty_constraint_sum_x_1(x) + penalty_constraint_fixed_reward(x,Returns,0.001)
res = bboptimize(my_new_fitness; SearchRange=(0,1),NumDimensions=N,TraceMode=:compact)
w_best = best_candidate(res)


res_GAP = bboptimize(my_GAP_fitness; SearchRange=(0,1),NumDimensions=N)
w = best_candidate(res_GAP)
GAP(w,Returns)
sum(w)
REWARD(w,Returns)
RISK(w,Returns)
penalty_constraint_fixed_reward(best_candidate(res),Returns,0.4)

# fitness_RISK(x) = (RISK(x,Returns),penalty_constraint_fixed_reward(x,Returns,0.4),penalty_constraint_sum_x_1(x))
# weightedFitness(f) = f[1]*0.02 + f[2]*0.49 + f[3]*0.49
# res = bboptimize(fitness_RISK; Method=:borg_moea,
#             FitnessScheme=ParetoFitnessScheme{3}(is_minimizing=true,aggregator=weightedFitness),
#             SearchRange=(0.0, 1.0), NumDimensions=N, ϵ=0.05,
#             MaxSteps=50000, TraceInterval=1.0, TraceMode=:compact);


w_best = best_candidate(res)
GAP(w_best,Returns)
# # 0.2571354403668967
# #  0.0009194161736479042
# #  0.7357003504270017
# #  0.0016068823425896975
# #  0.0046383085455844845

# w_uni = repeat([1/N], N)

# GAP(w_best,Returns)
# GAP([0.2571354403668967,0.0009194161736479042,0.7357003504270017,0.0016068823425896975,0.0046383085455844845], Returns)
# GAP(w_uni,Returns)

# a = rand(5)
# w_rand = a/sum(a)

# GAP(w_rand,Returns)