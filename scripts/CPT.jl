# cd("/home/nikap/Desktop/Masterthesis/Conic_Portfolio") # On Linux
using CSV
using DataFrames
using Statistics
using Plots
using IndependentComponentAnalysis
using Loess
using Distributions
using LinearAlgebra
using Optim
using Base.Threads: @threads, @spawn

include("ICA_assets.jl")
include("implied_moments.jl")
include("getVGParams.jl")
include("VG_Params2MRet.jl")
include("MINMAXVAR.jl")

println("Importing datasets")
names = ["BRK-B","NKE","FB","V","GOOGL"];
csv_brk = CSV.File("datasets/stocks/BRK-B.csv");
csv_nke = CSV.File("datasets/stocks/NKE.csv");
csv_fb = CSV.File("datasets/stocks/FB.csv");
csv_v = CSV.File("datasets/stocks/V.csv");
csv_googl = CSV.File("datasets/stocks/GOOGL.csv");

df_brk = DataFrame(csv_brk) # NIKE(NKE) GOOGLE(GOOGL) VISA(V) META(FB) BERKSHIRE(BRK-A)
df_nke = DataFrame(csv_nke)
df_fb = DataFrame(csv_fb)
df_v = DataFrame(csv_v)
df_googl = DataFrame(csv_googl)

# list of all stock dataframes.
df_list = [df_brk, df_nke, df_fb, df_v, df_googl];

# list of all stock dividend yields.
yields = [0, 0.0078, 0, 0.0069, 0];

nrAssets = length(df_list); #number of assets
sampleReturns = zeros(nrAssets,length(df_brk."Adj Close")-1) #matrix containing the daily returns of each asset in each row.
assetShiftedReturns = Array{Float64,2}(undef,nrAssets,length(df_brk."Adj Close")-1); #same as up here but then making mean = 0

for i = 1:nrAssets
    df = df_list[i]
    closes = df."Adj Close"
    returns = log.(closes[2:end]./closes[1:end-1])
    sampleReturns[i,:] = returns
    gemiddelde = mean(returns)
    shiftedReturns = returns-repeat([gemiddelde],length(returns))
    assetShiftedReturns[i,:] = shiftedReturns;
end

println("Starting Independent Component Analysis")
ica_comps = ICA_assets(assetShiftedReturns);

# X = A*S      where X are the returns, A is mixing matrix and S is independent comps
# S = W'*X
W = ica_comps.W;
A = ica_comps.mixing;
S = ica_comps.indcomps; #these are the independent components. For each row, we need σ, ν, θ. 

pars = get_params_timeseries_returns(S);

V = VG_Params2MRet([getVGParams(pars[1,:]...,1,20),
                getVGParams(pars[2,:]...,1,20),
                getVGParams(pars[3,:]...,1,20),
                getVGParams(pars[4,:]...,1,20),
                getVGParams(pars[5,:]...,1,20)], 10000)


#returns Y = A*V where A is the mixing matrix.
Y = A*V;
const N,M = size(Y);
Returns′ = zeros(N,M);
# Later on we also need an already sorted Returns matrix (sorted rows)
ReturnsSorted′ = zeros(N,M)
# Let us center the returns in Y 
for i = 1:N
    avgSumExp = 1/M*sum(exp.(Y[i,:]))
    # println(avgSumExp)
    for m = 1:M
        Returns′[i,m] = exp(Y[i,m])-avgSumExp
    end
    ReturnsSorted′[i,:] = sort(Returns′[i,:])
end

const Returns = Returns′
const ReturnsSorted = ReturnsSorted′
const Ψs = [MINMAXVAR(m/M,0.1) for m=0:M]
const ΨminusΨ = [Ψs[m]-Ψs[m-1] for m=2:(M+1)]

# Returns for the ordered vector R the distorted expectation defined by distortion Ψ
# result == ψ(m/M) - Ψ((m-1)/M)    where m = 1,...,M and M is length(R)
function bid(R::Vector{Float64})
    # ΨminusΨ = [Ψs[m]-Ψs[m-1] for m=1:M]
    # println("M = ", M, ",\nR = ",size(R),",\nΨminusΨ = ",size(ΨminusΨ))
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

# Reward μ_p = -∑ᵢ(aᵢ⋅bid(Rᵢ))
function REWARDSORTED(w::Vector{Float64},Rs::Matrix{Float64})
    μₚ = -w' * [    bid(Rs[1,:]),
                    bid(Rs[2,:]),
                    bid(Rs[3,:]),
                    bid(Rs[4,:]),
                    bid(Rs[5,:])]
    return μₚ
end

# Risk ̃c(a) = -bid(Rₚ)

#             w = n×1 vector     Rs = n×m matrix
function RISK(w::Vector{Float64},Rs::Matrix{Float64})
    Rₚ = Rs' * w
    c = -bid(sort(Rₚ))
    return c
end 

# using ProfileView: @profview #loading this makes everything so slow
# vektor = rand(5);vektor=vektor/sum(vektor);matriks=rand(5,1000);
# @profview (for i = 1:10000; RISK(vektor,matriks);end)

function GAP(w::Vector{Float64},Rs::Matrix{Float64})
    c = RISK(w,Rs)
    μₚ = REWARD(w,Rs)
    return μₚ - c
end

function GAPSORTED(w::Vector{Float64},Rs::Matrix{Float64},RsSorted::Matrix{Float64})
    c = RISK(w,Rs)
    μₚ = REWARD(w,RsSorted)
    return μₚ - c
end

# # optimization
# We need to know what the lowest reward is that we can get,
# and the highest reward that we can get. We can never be lower than
# the stock with the lowest return (as we are not shortselling)
# nor can we be higher than the stock with the highest reward
W_id = Matrix{Float64}(LinearAlgebra.I,N,N) # Identity matrix
rewardStocks = [REWARD(W_id[i,:],Returns) for i = 1:N]
I = range(minimum(rewardStocks);stop=maximum(rewardStocks),length=20)
ws = zeros(length(I),N)

# The following optimizes for the best gap.
println("Begin optimization:")
println("Optimize weights for gaps -")

x_0 = repeat([1/N], N);
lx = zeros(N); ux = ones(N);
lc = [0]; uc = [0];
fun(x) = -GAPSORTED(x,Returns,ReturnsSorted)
df = TwiceDifferentiable(fun, x_0)
con_c!(c, x) = (c[1] = sum(x)-1; c)
dfc = TwiceDifferentiableConstraints(con_c!, lx, ux, lc, uc)

res = optimize(df, dfc, x_0, IPNewton())

print("Best weights: ", Optim.minimizer(res), "\nwith gap: ", -Optim.minimum(res), "\n")
w_optimgap = Optim.minimizer(res)
# w_optimgap = [0.19906163286879, 0.32840144072909444, 0.32439569300690163, 0.06899830811045338, 0.0791429252847606]
# ↑↑ comment this out if you are not working with the test set ↑↑
gap_optim = -Optim.minimum(res)

funRISK(x) = RISK(x,Returns)

# @threads for i = 1:length(I)
@threads for i = 1:20
    println("Thread number ",Threads.threadid()," working on iteration ",i,"/",length(I))
    μₚ = I[i]
    # x = [1/2, 1/2]
    # c[1] = sum(x)-1 == 1 - 1 == 0 
    # c[2] = REWARDSORTED(x,ReturnsSorted) - μₚ
    function con_c!(c, x) 
        c[1] = sum(x)-1
        c[2] = REWARDSORTED(x,ReturnsSorted)-μₚ
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

# Begin the block here:
begin
    plot(   [RISK(ws[i,:],Returns) for i = 1:length(I)], 
            [REWARD(ws[i,:],Returns) for i = 1:length(I)],
            linecolor=:blue,
            label="Conic Efficient Frontier",
            title="Efficient Frontier",
            lw=3    )
    plot!(  [0,1],
            [gap_optim, gap_optim+1],
            label="Conic max diversification line",
            lw=3    )
    plot!(  [RISK(w_optimgap,Returns)],
            [REWARD(w_optimgap,Returns)],
            seriestype=:scatter,
            label="Max diversified portfolio",
            markershape=:diamond,
            markersize=6    )
    # plot!(  [RISK(w_optimMPT,Returns)],
            # [REWARD(w_optimMPT,Returns)],
            # seriestype=:scatter,
            # label="Minimum variance portfolio",
            # markershape=:star4,
            # markersize=6    )
    # plot!(  [RISK(w_optimSharpe,Returns)],
    #         [REWARD(w_optimSharpe,Returns)],
    #         seriestype=:scatter,
    #         label="Maximum Sharpe Portfolio",
    #         markershape=:star5,
    #         markersize=6    )
    # ↑↑ comment this out if you haven't calculated the weights through MPT optimization
    W_id = Matrix{Float64}(LinearAlgebra.I,N,N)
    plot!(  [RISK(W_id[i,:],Returns) for i = [1:3;5:N]], 
            [REWARD(W_id[i,:],Returns) for i = [1:3;5:N]],
            seriestype = :scatter,
            label="Stocks",
            series_annotations=text.(names[[1:3;5:N]], :top,:left;rotation=-20.2,pointsize=8),
            markercolor=:purple )
    plot!(  [RISK(W_id[4,:],Returns)], 
            [REWARD(W_id[4,:],Returns)],
            seriestype = :scatter,
            label=false,
            series_annotations=text.(names[4], :bottom,:right;rotation=-20.2,pointsize=8),
            markercolor=:purple )
    sample_ports = [(a=rand(N);a=a/sum(a);a) for i = 1:5]
    plot!(  [RISK(sample_ports[i],Returns) for i = 1:5],
            [REWARD(sample_ports[i],Returns) for i = 1:5],
            seriestype=:scatter,
            label="Randomly assembled portfolios",
            markeralpha=0.5,
            markerstrokealpha=0.5   )
    plot!(title="Conic Efficient Frontier",xlabel="Risk c̃(a)",ylabel="Reward μₚ")
    plot!(legend=:bottomright)
    #γ=0.1
    plot!(xlims=(0,0.0167),ylims=(0,0.02))
    #γ=0.8
    # plot!(xlims=(0,0.03),ylims=(0,0.035))
end
# End the block here ↑


## Long-Short portfolio:
W_id = Matrix{Float64}(LinearAlgebra.I,N,N) # Identity matrix
rewardStocks = [REWARD(W_id[i,:],Returns) for i = 1:N]
δ_reward = minimum(rewardStocks)/1.5 
I2 = range(δ_reward; stop=δ_reward+maximum(rewardStocks),length=20)
ws2 = zeros(length(I2),N)

Σ_indcomp_1m = cov(Returns') #monthly
Σ_sample_1d = cov(sampleReturns') #daily

# √(x_0'*Σ_indcomp_1m*x_0*12)

Vols = zeros(1000000)
for i = 1:length(Vols)
    # weights = (rand(N).-0.5).*1000
    weights = rand(N)
    weights = weights/sum(weights)
    Vols[i] = sqrt(weights'*Σ_indcomp_1m*weights)
end
histogram(log.(Vols*√(12));normalize=:probability,title="Histogram Log 1Y Volatility (Long-Short)",xlabel="σ",legend=false)
histogram((Vols*√(12));normalize=:probability,title="Histogram 1Y Volatility (Long)",xlabel="σ",legend=false)
Q = quantile(Vols*√(12),0.95)

begin
    println("Begin optimization (Long-Short):")
    println("Optimize weights for gaps -")

    fun(x) = -GAP(x,Returns)
    x_0 = w_optimgap
    df = TwiceDifferentiable(fun, x_0)

    con_c!(c, x) = (c[1] = sum(x)-1; c[2] = sqrt(x'*Σ_indcomp_1m*x); c)
    lx = fill(-Inf,N); ux = fill(Inf,N);
    lc = [0,0]; uc = [0,Q/√(12)];
    dfc = TwiceDifferentiableConstraints(con_c!, lx, ux, lc, uc)

    res2 = optimize(df, dfc,x_0, IPNewton())
    print("Best weights: ", Optim.minimizer(res2), "\nwith gap: ", -Optim.minimum(res2), "\n")
    w_optimgap2 = Optim.minimizer(res2)
    gap_optim2 = GAP(w_optimgap2,Returns)
end

begin
    funRISK(x) = RISK(x,Returns)

    @threads for i = 1:length(I2)
        println("Thread number ",Threads.threadid()," working on iteration ",i,"/",length(I2))
        μₚ = I2[i]
        function con_c!(c, x)
            c[1] = sum(x)-1
            c[2] = REWARD(x,Returns)-μₚ
            c[3] = sqrt(x'*Σ_indcomp_1m*x)
            c
        end
        lc = [0,0,0]; uc = [0,0,Q*1.5/√(12)];

        x_0 = ws[i,:]

        df = TwiceDifferentiable(funRISK, x_0)
        dfc = TwiceDifferentiableConstraints(con_c!, lx, ux, lc, uc)

        res = optimize(df, dfc, x_0, IPNewton())
        w = Optim.minimizer(res)
        ws2[i,:] = w
    end
end

begin
    bitvek = (([sum(ws2[i,:]) for i = 1:20].>0.9) .* ([sum(ws2[i,:]) for i = 1:20].<1.1))
    ws2 = ws2[bitvek,:]
    plot([RISK(ws2[i,:],Returns) for i = 1:size(ws2,1)], [REWARD(ws2[i,:],Returns) for i = 1:size(ws2,1)],linecolor=:blue,label="Conic Efficient Frontier",title="Efficient Frontier",lw=3)
    plot!([0,1],[gap_optim2, gap_optim2+1],label="Conic max diversification line",lw=3)
    plot!([RISK(w_optimgap2,Returns)],[REWARD(w_optimgap2,Returns)],seriestype=:scatter,label="Max diversified portfolio",markershape=:diamond,markersize=6)
    # plot!([RISK(w_optimMPT,Returns)],[REWARD(w_optimMPT,Returns)],seriestype=:scatter,label="Minimum variance portfolio",markershape=:star4,markersize=6)
    # ↑↑ comment this out if you haven't calculated the weights through MPT optimization
    W_id = Matrix{Float64}(LinearAlgebra.I,N,N)
    plot!(
        [RISK(W_id[i,:],Returns) for i = [1:3;5:N]], 
        [REWARD(W_id[i,:],Returns) for i = [1:3;5:N]],
        seriestype = :scatter,label="Stocks",
        series_annotations=text.(names[[1:3;5:N]], :top,:left;rotation=-20.2,pointsize=8),
        markercolor=:purple)
    plot!(
        [RISK(W_id[4,:],Returns)], 
        [REWARD(W_id[4,:],Returns)],
        seriestype = :scatter,
        label=false,
        series_annotations=text.(names[4], :bottom,:right;rotation=-20.2,pointsize=8),
        markercolor=:purple)
    sample_ports = [(a=rand(N);a=a/sum(a);a) for i = 1:5]
    plot!([RISK(sample_ports[i],Returns) for i = 1:5],[REWARD(sample_ports[i],Returns) for i = 1:5],seriestype=:scatter,label="Randomly assembled portfolios",markeralpha=0.5,markerstrokealpha=0.5)
    plot!(title="Conic Efficient Frontier",xlabel="Risk c̃(a)",ylabel="Reward μₚ")
    plot!(legend=:bottomright)
    #γ=0.1
    plot!(xlims=(0,0.02),ylims=(0,0.02))
    #γ=0.8
    # plot!(xlims=(0,0.03),ylims=(0,0.035))
end