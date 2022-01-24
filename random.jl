cd("/home/nikap/Desktop/Masterthesis/Conic_Portfolio") # On Linux
using CSV
using DataFrames
using Statistics
using Plots
using IndependentComponentAnalysis
using Loess
using Distributions
using LinearAlgebra
using Optim

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
        Δt = params[i].Δt
        g = Gamma(Δt*1/ν,1/ν)
        g_n = rand(g,M)
        ϵ = Normal()
        ϵ_n = rand(ϵ,M)
        for j = 1:M
            returns[i,j] = θ*(g_n[j]-1)+σ*√(g_n[j])*ϵ_n[j]
        end
    end
    return returns
end

V = VG_IndComp2MRet([getVGParams(pars[1,:]...,1,20),
                getVGParams(pars[2,:]...,1,20),
                getVGParams(pars[3,:]...,1,20),
                getVGParams(pars[4,:]...,1,20),
                getVGParams(pars[5,:]...,1,20)], 10000)

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

I = 0.01:0.002:0.0200
ws = zeros(length(I),N)

println("Begin optimization:")
println("Optimize weights for gaps -")

fun(x) = -GAP(x,Returns)
x_0 = repeat([1/N], N);
df = TwiceDifferentiable(fun, x_0)

con_c!(c, x) = (c[1] = sum(x)-1; c)
lx = zeros(N); ux = ones(N);
lc = [0]; uc = [0];
dfc = TwiceDifferentiableConstraints(con_c!, lx, ux, lc, uc)

res = optimize(df, dfc, x_0, IPNewton())
print("Best weights: ", Optim.minimizer(res), "\nwith gap: ", -Optim.minimum(res), "\n")
# w = Optim.minimizer(res)
w_optimgap = [0.1064360570943749, 0.8935639429056227, 7.633801505797622e-16, 1.1489027579482007e-15, 6.215419397489384e-16]
gap_optim = -Optim.minimum(res)
# plot([RISK(w_optimgap,Returns)],[REWARD(w_optimgap,Returns)],seriestype=:scatter)
plot!([RISK(w_optimgap,Returns)],[REWARD(w_optimgap,Returns)],seriestype=:scatter)
plot!(xlims=(0.00,0.02),ylims=(0.00,0.04))
plot!([0,0.03],[gap_optim, gap_optim+0.03])

funRISK(x) = RISK(x,Returns)
println(1)
μₚ = I[1]
function con_c!(c, x)
    c[1] = sum(x)-1
    c[2] = REWARD(x,Returns)-μₚ
    c
end
lc = [0,0]; uc = [0,0];
x0 = rand(N)
x0 = x0/sum(x0)
df = TwiceDifferentiable(funRISK, x0)
dfc = TwiceDifferentiableConstraints(con_c!, lx, ux, lc, uc)
res = optimize(df, dfc, x0, IPNewton())
w = Optim.minimizer(res)
plot!([RISK(w,Returns)],[REWARD(w,Returns)],seriestype=:scatter)
ws[1,:] = w

for i = 2:length(I)
    println(i)
    μₚ = I[i]
    function con_c!(c, x)
        c[1] = sum(x)-1
        c[2] = REWARD(x,Returns)-μₚ
        c
    end
    lc = [0,0]; uc = [0,0];
    x0 = rand(N)
    x0 = x0/sum(x0)
    df = TwiceDifferentiable(funRISK, x0)
    dfc = TwiceDifferentiableConstraints(con_c!, lx, ux, lc, uc)

    res = optimize(df, dfc, x0, IPNewton())
    w = Optim.minimizer(res)
    ws[i,:] = w
end
plot!(legend=false)
plot!([RISK(ws[i,:],Returns) for i = 1:length(I)], [REWARD(ws[i,:],Returns) for i = 1:length(I)],seriestype = :scatter,linecolor=:blue)

I2 = 0.022:0.002:0.040
ws2 = zeros(length(I2),N)
for i = 1:length(I2)
    println("iteration ", i, " out of ",length(I2))
    μₚ = I2[i]
    function con_c!(c, x)
        c[1] = sum(x)-1
        c[2] = REWARD(x,Returns)-μₚ
        c
    end
    lc = [0,0]; uc = [0,0];
    x0 = rand(N)
    x0 = x0/sum(x0)
    df = TwiceDifferentiable(funRISK, x0)
    dfc = TwiceDifferentiableConstraints(con_c!, lx, ux, lc, uc)

    res = optimize(df, dfc, x0, IPNewton())
    w = Optim.minimizer(res)
    ws2[i,:] = w
end
plot!([RISK(ws2[i,:],Returns) for i = 1:length(I2)], [REWARD(ws2[i,:],Returns) for i = 1:length(I2)],seriestype = :scatter,linecolor=:blue)

CSV.write("OPT_w.txt", DataFrame(ws,:auto),header=false);
CSV.write("OPT_Y.txt", DataFrame(Y,:auto),header=false);
CSV.write("OPT_A.txt", DataFrame(A,:auto),header=false);
CSV.write("OPT_V.txt", DataFrame(V,:auto),header=false);
CSV.write("OPT_RETURNS.txt", DataFrame(Returns,:auto),header=false);
CSV.write("OPT_pointsxy.txt", DataFrame([pointsx pointsy],:auto),header=false);
CSV.write("OPT_ws.txt", DataFrame(ws2,:auto),header=false);
CSV.write("OPT_w.txt", DataFrame([w_optimgap w_optimMPT],:auto),header=false);

plot(pointsx,pointsy,linewidth=3,label="Conic Efficient Frontier")
plot!([RISK(w_optimgap,Returns)],[REWARD(w_optimgap,Returns)],seriestype=:scatter,label="Max diversified portfolio")
plot!([0,1],[gap_optim, gap_optim+1],label="Conic max diversification line")

W_id = Matrix{Float64}(LinearAlgebra.I,N,N)
plot!([RISK(W_id[i,:],Returns) for i = 1:N], [REWARD(W_id[i,:],Returns) for i = 1:N],seriestype = :scatter,label="Stocks")
plot!(xlims=(0.0,0.02),ylims=(0.0,0.04))
plot!(legend=:right)
plot!(title="Conic Efficient Frontier",xlabel="Risk c̃(a)",ylabel="Reward μₚ")

sample_ports = [(a=rand(N);a=a/sum(a);a) for i = 1:10]
plot!([RISK(sample_ports[i],Returns) for i = 1:10],[REWARD(sample_ports[i],Returns) for i = 1:10],seriestype=:scatter,label="Sample portfolios")


plot([RISK(w_optimgap,Returns)],[REWARD(w_optimgap,Returns)],seriestype=:scatter,label="Max diversified portfolio")
plot!([0,1],[gap_optim, gap_optim+1],label="Conic max diversification line")
plot!([RISK(w_optimMPT,Returns)],[REWARD(w_optimMPT,Returns)],seriestype=:scatter,label="Minimum variance portfolio")
plot!(legend=:right)
plot!(xlims=(0,0.15),ylims=(0,0.15))

μₚ = REWARD(w_optimMPT,Returns)
function con_c!(c, x)
    c[1] = sum(x)-1
    c[2] = REWARD(x,Returns)-μₚ
    c
end
lc = [0,0]; uc = [0,0];
x0 = rand(N)
x0 = x0/sum(x0)
df = TwiceDifferentiable(funRISK, x0)
dfc = TwiceDifferentiableConstraints(con_c!, lx, ux, lc, uc)
res = optimize(df, dfc, x0, IPNewton())